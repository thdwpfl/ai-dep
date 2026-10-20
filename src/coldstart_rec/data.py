"""MovieLens 1M download, preprocessing, and cold-start episode utilities.

The processed representation is intentionally user-ID-embedding free.  Ratings
are stored as timestamp-sorted CSR histories, while user-disjoint split labels
make it straightforward to construct genuine cold-start episodes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import urllib.parse
import urllib.request
import zipfile
from array import array
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RAW_FILENAMES = ("users.dat", "movies.dat", "ratings.dat")
ARTIFACT_FILENAME = "movielens_1m.npz"
MANIFEST_FILENAME = "split_manifest.json"
SCHEMA_VERSION = 1

# Canonical MovieLens 1M genre vocabulary (exactly 18 columns).
GENRES: tuple[str, ...] = (
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)
GENRE_TO_INDEX: Mapping[str, int] = {genre: index for index, genre in enumerate(GENRES)}

# U is retained for inference even though MovieLens 1M contains only M/F.
GENDER_LABELS: tuple[str, ...] = ("M", "F", "U")
GENDER_TO_INDEX: Mapping[str, int] = {label: index for index, label in enumerate(GENDER_LABELS)}
UNKNOWN_GENDER_INDEX = GENDER_TO_INDEX["U"]

SPLIT_LABELS: tuple[str, ...] = ("train", "val", "test")
SPLIT_TO_INDEX: Mapping[str, int] = {label: index for index, label in enumerate(SPLIT_LABELS)}

_YEAR_PATTERN = re.compile(r"\s*\((\d{4})\)\s*$")


@dataclass(frozen=True, slots=True)
class UserRecord:
    """One row from MovieLens ``users.dat``."""

    user_id: int
    gender: str
    age: int
    occupation: int
    zip_code: str


@dataclass(frozen=True, slots=True)
class MovieRecord:
    """One parsed row from MovieLens ``movies.dat``."""

    movie_id: int
    title: str
    year: int
    genre_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RatingRecord:
    """One row from MovieLens ``ratings.dat``."""

    user_id: int
    movie_id: int
    rating: int
    timestamp: int


@dataclass(frozen=True)
class UserHistory:
    """A single user's complete chronological rating history."""

    user_index: int
    user_id: int
    gender_index: int
    item_indices: NDArray[np.int32]
    movie_ids: NDArray[np.int32]
    ratings: NDArray[np.float32]
    timestamps: NDArray[np.int64]


@dataclass(frozen=True)
class ColdStartEpisode:
    """Chronological support/query episode for an unseen user."""

    user_index: int
    user_id: int
    gender_index: int
    gender_features: NDArray[np.float32]
    support_item_indices: NDArray[np.int32]
    support_movie_ids: NDArray[np.int32]
    support_ratings: NDArray[np.float32]
    support_timestamps: NDArray[np.int64]
    support_genres: NDArray[np.float32]
    query_item_indices: NDArray[np.int32]
    query_movie_ids: NDArray[np.int32]
    query_ratings: NDArray[np.float32]
    query_timestamps: NDArray[np.int64]
    query_genres: NDArray[np.float32]


def normalize_gender(value: str | None) -> int:
    """Map UI/user input to M/F/U, retaining an explicit unknown category."""

    if value is None:
        return UNKNOWN_GENDER_INDEX
    normalized = str(value).strip().casefold().replace("_", " ").replace("-", " ")
    aliases = {
        "m": "M",
        "male": "M",
        "man": "M",
        "f": "F",
        "female": "F",
        "woman": "F",
        "": "U",
        "u": "U",
        "unknown": "U",
        "other": "U",
        "prefer not to answer": "U",
        "prefer not to say": "U",
        "unspecified": "U",
    }
    try:
        return GENDER_TO_INDEX[aliases[normalized]]
    except KeyError as exc:
        raise ValueError(f"Unsupported gender value: {value!r}") from exc


def gender_one_hot(value: str | int | None) -> NDArray[np.float32]:
    """Return a three-column M/F/U inference feature."""

    index = int(value) if isinstance(value, (int, np.integer)) else normalize_gender(value)
    if not 0 <= index < len(GENDER_LABELS):
        raise ValueError(f"Gender index must be in [0, {len(GENDER_LABELS) - 1}], got {index}")
    result = np.zeros(len(GENDER_LABELS), dtype=np.float32)
    result[index] = 1.0
    return result


def split_title_year(raw_title: str) -> tuple[str, int]:
    """Split the terminal ``(YYYY)`` suffix while preserving title punctuation."""

    match = _YEAR_PATTERN.search(raw_title)
    if match is None:
        return raw_title.strip(), -1
    return raw_title[: match.start()].strip(), int(match.group(1))


def _iter_dat_rows(path: str | Path, expected_fields: int) -> Iterator[tuple[int, list[str]]]:
    file_path = Path(path)
    with file_path.open("r", encoding="latin-1", newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("::", maxsplit=expected_fields - 1)
            if len(fields) != expected_fields:
                raise ValueError(
                    f"{file_path}:{line_number}: expected {expected_fields} fields, "
                    f"found {len(fields)}"
                )
            yield line_number, fields


def iter_users(path: str | Path) -> Iterator[UserRecord]:
    """Yield validated user records from a latin-1 ``users.dat`` file."""

    for line_number, fields in _iter_dat_rows(path, 5):
        try:
            gender = fields[1].strip().upper()
            if gender not in ("M", "F"):
                raise ValueError(f"invalid MovieLens gender {gender!r}")
            yield UserRecord(
                user_id=int(fields[0]),
                gender=gender,
                age=int(fields[2]),
                occupation=int(fields[3]),
                zip_code=fields[4],
            )
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc


def iter_movies(path: str | Path) -> Iterator[MovieRecord]:
    """Yield movies with a clean title/year and canonical genre indices."""

    for line_number, fields in _iter_dat_rows(path, 3):
        title, year = split_title_year(fields[1])
        genre_names = fields[2].split("|")
        unknown = sorted(set(genre_names).difference(GENRE_TO_INDEX))
        if unknown:
            raise ValueError(f"{path}:{line_number}: unknown genres: {unknown}")
        yield MovieRecord(
            movie_id=int(fields[0]),
            title=title,
            year=year,
            genre_indices=tuple(GENRE_TO_INDEX[name] for name in genre_names),
        )


def iter_ratings(path: str | Path) -> Iterator[RatingRecord]:
    """Yield validated rating rows."""

    for line_number, fields in _iter_dat_rows(path, 4):
        try:
            rating = int(fields[2])
            if rating < 1 or rating > 5:
                raise ValueError(f"rating must be 1..5, got {rating}")
            timestamp = int(fields[3])
            if timestamp < 0:
                raise ValueError(f"timestamp must be non-negative, got {timestamp}")
            yield RatingRecord(
                user_id=int(fields[0]),
                movie_id=int(fields[1]),
                rating=rating,
                timestamp=timestamp,
            )
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc


def _sha256(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_raw_directory(path: Path) -> bool:
    return path.is_dir() and all((path / filename).is_file() for filename in RAW_FILENAMES)


def resolve_raw_directory(path: str | Path) -> Path:
    """Resolve either ``.../raw`` or ``.../raw/ml-1m`` to the dataset directory."""

    candidate = Path(path).expanduser().resolve()
    if _validate_raw_directory(candidate):
        return candidate
    nested = candidate / "ml-1m"
    if _validate_raw_directory(nested):
        return nested
    raise FileNotFoundError(
        f"Could not find {', '.join(RAW_FILENAMES)} in {candidate} or {nested}"
    )


def safe_extract_zip(
    archive_path: str | Path,
    destination: str | Path,
    *,
    max_uncompressed_bytes: int = 2_000_000_000,
) -> Path:
    """Extract a ZIP after rejecting traversal, links, encryption, and huge payloads."""

    archive = Path(archive_path)
    target_root = Path(destination)
    target_root.mkdir(parents=True, exist_ok=True)
    resolved_root = target_root.resolve()
    seen_files: set[Path] = set()

    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        total_size = sum(info.file_size for info in infos)
        if total_size > max_uncompressed_bytes:
            raise ValueError(
                f"Archive expands to {total_size} bytes; limit is {max_uncompressed_bytes}"
            )
        for info in infos:
            name = info.filename
            pure_path = PurePosixPath(name)
            unix_mode = info.external_attr >> 16
            if (
                not name
                or pure_path.is_absolute()
                or "\\" in name
                or ".." in pure_path.parts
                or stat.S_ISLNK(unix_mode)
                or info.flag_bits & 0x1
            ):
                raise ValueError(f"Unsafe ZIP member: {name!r}")
            output_path = (target_root.joinpath(*pure_path.parts)).resolve()
            if os.path.commonpath((str(resolved_root), str(output_path))) != str(resolved_root):
                raise ValueError(f"ZIP member escapes destination: {name!r}")
            if info.is_dir():
                output_path.mkdir(parents=True, exist_ok=True)
                continue
            if output_path in seen_files:
                raise ValueError(f"Duplicate ZIP member: {name!r}")
            seen_files.add(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info, "r") as source, output_path.open("xb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
    return target_root


def _download_file(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None,
    timeout_seconds: int,
    max_download_bytes: int,
) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("MovieLens downloads must use HTTPS")
    request = urllib.request.Request(url, headers={"User-Agent": "coldstart-rec/1.0"})
    fd, temporary_name = tempfile.mkstemp(prefix=".ml1m-", suffix=".zip", dir=destination.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        downloaded = 0
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            final_scheme = urllib.parse.urlparse(response.geturl()).scheme
            if final_scheme != "https":
                raise ValueError("Refusing a download redirected away from HTTPS")
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_download_bytes:
                raise ValueError("MovieLens archive exceeds configured download limit")
            with temporary_path.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > max_download_bytes:
                        raise ValueError("MovieLens archive exceeds configured download limit")
                    output.write(chunk)
        if expected_sha256 and _sha256(temporary_path) != expected_sha256.lower():
            raise ValueError("Downloaded MovieLens archive SHA-256 mismatch")
        with zipfile.ZipFile(temporary_path) as bundle:
            corrupt_member = bundle.testzip()
        if corrupt_member is not None:
            raise ValueError(f"Corrupt ZIP member: {corrupt_member}")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def download_ml1m(
    raw_root: str | Path,
    *,
    url: str = ML1M_URL,
    expected_sha256: str | None = None,
    force: bool = False,
    timeout_seconds: int = 60,
    max_download_bytes: int = 1_000_000_000,
) -> Path:
    """Download and safely extract MovieLens 1M, returning ``.../ml-1m``."""

    root = Path(raw_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset_path = root / "ml-1m"
    if _validate_raw_directory(dataset_path) and not force:
        return dataset_path

    archive_path = root / "ml-1m.zip"
    archive_is_usable = archive_path.is_file()
    if archive_is_usable and expected_sha256:
        archive_is_usable = _sha256(archive_path) == expected_sha256.lower()
    if archive_is_usable:
        try:
            with zipfile.ZipFile(archive_path) as bundle:
                archive_is_usable = bundle.testzip() is None
        except zipfile.BadZipFile:
            archive_is_usable = False
    if force or not archive_is_usable:
        _download_file(
            url,
            archive_path,
            expected_sha256=expected_sha256,
            timeout_seconds=timeout_seconds,
            max_download_bytes=max_download_bytes,
        )

    temporary_root = Path(tempfile.mkdtemp(prefix=".ml1m-extract-", dir=root))
    backup_path: Path | None = None
    try:
        safe_extract_zip(archive_path, temporary_root)
        extracted_path = resolve_raw_directory(temporary_root)
        if dataset_path.exists():
            backup_path = root / f".ml-1m-backup-{os.getpid()}"
            if backup_path.exists():
                raise FileExistsError(f"Refusing to overwrite backup path {backup_path}")
            os.replace(dataset_path, backup_path)
        os.replace(extracted_path, dataset_path)
        if backup_path is not None:
            shutil.rmtree(backup_path)
            backup_path = None
    except Exception:
        if backup_path is not None and backup_path.exists() and not dataset_path.exists():
            os.replace(backup_path, dataset_path)
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return dataset_path


def _validate_split_ratios(split_ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(split_ratios) != 3:
        raise ValueError("split_ratios must contain train, validation, and test ratios")
    ratios = tuple(float(value) for value in split_ratios)
    if any(not math.isfinite(value) or value <= 0 for value in ratios):
        raise ValueError("All split ratios must be finite and positive")
    total = sum(ratios)
    return tuple(value / total for value in ratios)  # type: ignore[return-value]


def _allocate_split_counts(size: int, ratios: Sequence[float]) -> NDArray[np.int64]:
    exact = np.asarray(ratios, dtype=np.float64) * size
    counts = np.floor(exact).astype(np.int64)
    remainder = size - int(counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[:remainder]] += 1
    if size >= len(ratios):
        for empty_index in np.flatnonzero(counts == 0):
            donors = np.flatnonzero(counts > 1)
            donor = donors[np.argmax(counts[donors] - exact[donors])]
            counts[donor] -= 1
            counts[empty_index] += 1
    return counts


def stratified_user_split(
    gender_indices: NDArray[np.integer[Any]],
    *,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> NDArray[np.uint8]:
    """Assign each user to exactly one split, stratified by gender."""

    ratios = _validate_split_ratios(split_ratios)
    genders = np.asarray(gender_indices)
    labels = np.full(genders.shape[0], 255, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for gender in np.unique(genders):
        indices = np.flatnonzero(genders == gender)
        shuffled = rng.permutation(indices)
        counts = _allocate_split_counts(len(indices), ratios)
        start = 0
        for split_index, count in enumerate(counts):
            stop = start + int(count)
            labels[shuffled[start:stop]] = split_index
            start = stop
    if np.any(labels == 255):
        raise RuntimeError("Internal error: some users were not assigned to a split")
    return labels


def _atomic_save_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_save_json(path: Path, value: Mapping[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def preprocess_ml1m(
    raw_directory: str | Path,
    output_directory: str | Path,
    *,
    seed: int = 42,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
) -> tuple[Path, Path]:
    """Parse MovieLens 1M and atomically write an NPZ artifact and split manifest."""

    raw_path = resolve_raw_directory(raw_directory)
    output_path = Path(output_directory).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    ratios = _validate_split_ratios(split_ratios)

    users = sorted(iter_users(raw_path / "users.dat"), key=lambda row: row.user_id)
    movies = sorted(iter_movies(raw_path / "movies.dat"), key=lambda row: row.movie_id)
    if len({row.user_id for row in users}) != len(users):
        raise ValueError("Duplicate user IDs in users.dat")
    if len({row.movie_id for row in movies}) != len(movies):
        raise ValueError("Duplicate movie IDs in movies.dat")

    user_ids = np.asarray([row.user_id for row in users], dtype=np.int32)
    user_genders = np.asarray([GENDER_TO_INDEX[row.gender] for row in users], dtype=np.uint8)
    user_ages = np.asarray([row.age for row in users], dtype=np.int16)
    user_occupations = np.asarray([row.occupation for row in users], dtype=np.int16)
    movie_ids = np.asarray([row.movie_id for row in movies], dtype=np.int32)
    movie_titles = np.asarray([row.title for row in movies], dtype=np.str_)
    movie_years = np.asarray([row.year for row in movies], dtype=np.int16)
    movie_genres = np.zeros((len(movies), len(GENRES)), dtype=np.uint8)
    for movie_index, movie in enumerate(movies):
        movie_genres[movie_index, list(movie.genre_indices)] = 1

    valid_years = movie_years[movie_years >= 0]
    year_center = float(np.mean(valid_years)) if valid_years.size else 0.0
    year_scale = float(np.std(valid_years)) if valid_years.size else 1.0
    if year_scale < 1e-8:
        year_scale = 1.0
    movie_years_normalized = np.where(
        movie_years >= 0, (movie_years.astype(np.float32) - year_center) / year_scale, 0.0
    ).astype(np.float32)

    user_lookup = {int(user_id): index for index, user_id in enumerate(user_ids)}
    movie_lookup = {int(movie_id): index for index, movie_id in enumerate(movie_ids)}
    rating_user_indices = array("i")
    rating_item_indices_buffer = array("i")
    rating_values_buffer = array("B")
    rating_timestamps_buffer = array("q")
    for rating in iter_ratings(raw_path / "ratings.dat"):
        try:
            user_index = user_lookup[rating.user_id]
            movie_index = movie_lookup[rating.movie_id]
        except KeyError as exc:
            raise ValueError(f"Rating references unknown ID: {exc.args[0]}") from exc
        rating_user_indices.append(user_index)
        rating_item_indices_buffer.append(movie_index)
        rating_values_buffer.append(rating.rating)
        rating_timestamps_buffer.append(rating.timestamp)

    rating_users = np.asarray(rating_user_indices, dtype=np.int32)
    rating_items = np.asarray(rating_item_indices_buffer, dtype=np.int32)
    rating_values = np.asarray(rating_values_buffer, dtype=np.uint8)
    rating_timestamps = np.asarray(rating_timestamps_buffer, dtype=np.int64)
    input_order = np.arange(len(rating_values), dtype=np.int64)
    order = np.lexsort((input_order, rating_timestamps, rating_users))
    rating_users = rating_users[order]
    rating_items = rating_items[order]
    rating_values = rating_values[order]
    rating_timestamps = rating_timestamps[order]

    history_counts = np.bincount(rating_users, minlength=len(users)).astype(np.int64)
    if np.any(history_counts == 0):
        missing = user_ids[history_counts == 0].tolist()
        raise ValueError(f"Users without ratings are unsupported; examples: {missing[:10]}")
    user_ptr = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(history_counts)))
    user_splits = stratified_user_split(user_genders, split_ratios=ratios, seed=seed)
    # Long-form item-group membership used directly by DEE estimators.
    dee_movie_indices, dee_genre_indices = np.nonzero(movie_genres)
    dee_item_genre_membership = np.column_stack(
        (dee_movie_indices.astype(np.int32), dee_genre_indices.astype(np.int16))
    ).astype(np.int32)

    artifact_path = output_path / ARTIFACT_FILENAME
    arrays: dict[str, NDArray[Any]] = {
        "movie_ids": movie_ids,
        "movie_titles": movie_titles,
        "movie_years": movie_years,
        "movie_years_normalized": movie_years_normalized,
        "movie_genres": movie_genres,
        "dee_item_genre_membership": dee_item_genre_membership,
        "user_ids": user_ids,
        "user_genders": user_genders,
        "user_ages": user_ages,
        "user_occupations": user_occupations,
        "user_splits": user_splits,
        "user_ptr": user_ptr,
        "rating_item_indices": rating_items,
        "rating_values": rating_values,
        "rating_timestamps": rating_timestamps,
    }
    _atomic_save_npz(artifact_path, arrays)

    split_sections: dict[str, Any] = {}
    for split_index, split_name in enumerate(SPLIT_LABELS):
        indices = np.flatnonzero(user_splits == split_index)
        split_sections[split_name] = {
            "num_users": int(len(indices)),
            "num_ratings": int(history_counts[indices].sum()),
            "gender_user_counts": {
                label: int(np.count_nonzero(user_genders[indices] == gender_index))
                for gender_index, label in enumerate(GENDER_LABELS)
            },
            "user_ids": [int(value) for value in user_ids[indices]],
        }

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "MovieLens 1M",
        "source_url": ML1M_URL,
        "seed": int(seed),
        "split_ratios": dict(zip(SPLIT_LABELS, ratios)),
        "split_strategy": "user-disjoint, gender-stratified",
        "history_order": "timestamp ascending, then original row order",
        "no_user_overlap": True,
        "genres": list(GENRES),
        "gender_to_index": dict(GENDER_TO_INDEX),
        "unknown_gender_note": "U is reserved for inference; MovieLens 1M contains M/F only.",
        "year_normalization": {"center": year_center, "scale": year_scale, "missing_value": 0.0},
        "counts": {
            "users": len(users),
            "movies": len(movies),
            "ratings": len(rating_values),
            "genre_memberships": int(len(dee_item_genre_membership)),
            "history_min": int(history_counts.min()),
            "history_median": float(np.median(history_counts)),
            "history_max": int(history_counts.max()),
        },
        "raw_files": {
            filename: {
                "bytes": (raw_path / filename).stat().st_size,
                "sha256": _sha256(raw_path / filename),
            }
            for filename in RAW_FILENAMES
        },
        "artifact": {"filename": ARTIFACT_FILENAME, "sha256": _sha256(artifact_path)},
        "splits": split_sections,
    }
    manifest_path = output_path / MANIFEST_FILENAME
    _atomic_save_json(manifest_path, manifest)
    return artifact_path, manifest_path


class MovieLens1MData:
    """In-memory, validated view of the compact MovieLens artifact."""

    _REQUIRED_ARRAYS = (
        "movie_ids",
        "movie_titles",
        "movie_years",
        "movie_years_normalized",
        "movie_genres",
        "dee_item_genre_membership",
        "user_ids",
        "user_genders",
        "user_splits",
        "user_ptr",
        "rating_item_indices",
        "rating_values",
        "rating_timestamps",
    )

    def __init__(self, arrays: Mapping[str, NDArray[Any]], manifest: Mapping[str, Any]) -> None:
        for name in self._REQUIRED_ARRAYS:
            if name not in arrays:
                raise ValueError(f"Processed artifact is missing array {name!r}")
            setattr(self, name, arrays[name])
        self.user_ages = arrays.get("user_ages")
        self.user_occupations = arrays.get("user_occupations")
        self.manifest = dict(manifest)
        self._validate()
        self._user_id_to_index: dict[int, int] | None = None
        self._movie_id_to_index: dict[int, int] | None = None

    def _validate(self) -> None:
        if self.movie_genres.shape != (len(self.movie_ids), len(GENRES)):
            raise ValueError("movie_genres has an incompatible shape")
        if len(self.movie_years) != len(self.movie_ids):
            raise ValueError("movie_years length does not match movie_ids")
        if len(self.movie_years_normalized) != len(self.movie_ids):
            raise ValueError("movie_years_normalized length does not match movie_ids")
        if len(self.user_ptr) != len(self.user_ids) + 1 or int(self.user_ptr[0]) != 0:
            raise ValueError("user_ptr is not a valid CSR pointer")
        if np.any(np.diff(self.user_ptr) < 0):
            raise ValueError("user_ptr must be monotonically nondecreasing")
        rating_count = len(self.rating_values)
        if (
            int(self.user_ptr[-1]) != rating_count
            or len(self.rating_item_indices) != rating_count
            or len(self.rating_timestamps) != rating_count
        ):
            raise ValueError("Rating arrays have inconsistent lengths")
        if np.any(self.rating_item_indices < 0) or np.any(
            self.rating_item_indices >= len(self.movie_ids)
        ):
            raise ValueError("rating_item_indices contains an invalid item index")
        if np.any(self.user_splits >= len(SPLIT_LABELS)):
            raise ValueError("user_splits contains an invalid split")
        if np.any(self.user_genders >= len(GENDER_LABELS)):
            raise ValueError("user_genders contains an invalid gender")
        for user_index in range(len(self.user_ids)):
            start, stop = self.user_ptr[user_index : user_index + 2]
            if np.any(np.diff(self.rating_timestamps[int(start) : int(stop)]) < 0):
                raise ValueError(f"Ratings for user index {user_index} are not chronological")

    @property
    def num_users(self) -> int:
        return len(self.user_ids)

    @property
    def num_movies(self) -> int:
        return len(self.movie_ids)

    @property
    def num_ratings(self) -> int:
        return len(self.rating_values)

    @property
    def history_lengths(self) -> NDArray[np.int64]:
        return np.diff(self.user_ptr)

    @property
    def user_id_to_index(self) -> Mapping[int, int]:
        if self._user_id_to_index is None:
            self._user_id_to_index = {
                int(user_id): index for index, user_id in enumerate(self.user_ids)
            }
        return self._user_id_to_index

    @property
    def movie_id_to_index(self) -> Mapping[int, int]:
        if self._movie_id_to_index is None:
            self._movie_id_to_index = {
                int(movie_id): index for index, movie_id in enumerate(self.movie_ids)
            }
        return self._movie_id_to_index

    def split_user_indices(self, split: str) -> NDArray[np.int64]:
        """Return processed user indices belonging only to ``split``."""

        try:
            split_index = SPLIT_TO_INDEX[split.casefold()]
        except KeyError as exc:
            raise ValueError(f"Unknown split {split!r}; expected {SPLIT_LABELS}") from exc
        return np.flatnonzero(self.user_splits == split_index)

    def history_by_index(self, user_index: int) -> UserHistory:
        """Return one user's timestamp-sorted history by dense user index."""

        if not 0 <= user_index < self.num_users:
            raise IndexError(f"User index out of bounds: {user_index}")
        start, stop = (int(value) for value in self.user_ptr[user_index : user_index + 2])
        item_indices = self.rating_item_indices[start:stop].astype(np.int32, copy=False)
        return UserHistory(
            user_index=user_index,
            user_id=int(self.user_ids[user_index]),
            gender_index=int(self.user_genders[user_index]),
            item_indices=item_indices,
            movie_ids=self.movie_ids[item_indices].astype(np.int32, copy=False),
            ratings=self.rating_values[start:stop].astype(np.float32),
            timestamps=self.rating_timestamps[start:stop].astype(np.int64, copy=False),
        )

    def history_by_id(self, user_id: int) -> UserHistory:
        """Return one user's timestamp-sorted history by original MovieLens ID."""

        try:
            index = self.user_id_to_index[int(user_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown MovieLens user ID: {user_id}") from exc
        return self.history_by_index(index)

    def episode_by_index(self, user_index: int, support_size: int) -> ColdStartEpisode:
        """Use the first ``support_size`` ratings as support and all later ones as query."""

        if support_size < 0:
            raise ValueError("support_size must be non-negative")
        history = self.history_by_index(user_index)
        boundary = min(support_size, len(history.ratings))
        support_items = history.item_indices[:boundary]
        query_items = history.item_indices[boundary:]
        return ColdStartEpisode(
            user_index=user_index,
            user_id=history.user_id,
            gender_index=history.gender_index,
            gender_features=gender_one_hot(history.gender_index),
            support_item_indices=support_items,
            support_movie_ids=history.movie_ids[:boundary],
            support_ratings=history.ratings[:boundary],
            support_timestamps=history.timestamps[:boundary],
            support_genres=self.movie_genres[support_items].astype(np.float32),
            query_item_indices=query_items,
            query_movie_ids=history.movie_ids[boundary:],
            query_ratings=history.ratings[boundary:],
            query_timestamps=history.timestamps[boundary:],
            query_genres=self.movie_genres[query_items].astype(np.float32),
        )

    def search_movies(self, query: str, *, limit: int = 25) -> list[int]:
        """Return dense movie indices whose title contains ``query`` (case-insensitive)."""

        if limit < 1:
            return []
        needle = query.strip().casefold()
        if not needle:
            return list(range(min(limit, self.num_movies)))
        return [
            index
            for index, title in enumerate(self.movie_titles)
            if needle in str(title).casefold()
        ][:limit]


class ColdStartEpisodeDataset(Sequence[ColdStartEpisode]):
    """Sequence of chronological unseen-user episodes for a named user split."""

    def __init__(
        self,
        data: MovieLens1MData,
        split: str,
        support_size: int,
        *,
        min_query_size: int = 1,
    ) -> None:
        if support_size < 0 or min_query_size < 0:
            raise ValueError("support_size and min_query_size must be non-negative")
        self.data = data
        self.split = split
        self.support_size = support_size
        eligible = data.history_lengths >= support_size + min_query_size
        split_mask = data.user_splits == SPLIT_TO_INDEX.get(split.casefold(), 255)
        if not np.any(split_mask) and split.casefold() not in SPLIT_TO_INDEX:
            raise ValueError(f"Unknown split {split!r}; expected {SPLIT_LABELS}")
        self.user_indices = np.flatnonzero(eligible & split_mask)

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, index: int | slice) -> ColdStartEpisode | list[ColdStartEpisode]:
        if isinstance(index, slice):
            return [
                self.data.episode_by_index(int(user_index), self.support_size)
                for user_index in self.user_indices[index]
            ]
        return self.data.episode_by_index(int(self.user_indices[index]), self.support_size)


def load_processed_data(path: str | Path) -> MovieLens1MData:
    """Load a processed directory or NPZ file without enabling pickle."""

    requested = Path(path).expanduser().resolve()
    if requested.is_dir():
        artifact_path = requested / ARTIFACT_FILENAME
        manifest_path = requested / MANIFEST_FILENAME
    else:
        artifact_path = requested
        manifest_path = requested.with_name(MANIFEST_FILENAME)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Processed artifact not found: {artifact_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    with np.load(artifact_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version {manifest.get('schema_version')}; "
            f"expected {SCHEMA_VERSION}"
        )
    return MovieLens1MData(arrays, manifest)


__all__ = [
    "ARTIFACT_FILENAME",
    "ColdStartEpisode",
    "ColdStartEpisodeDataset",
    "GENRES",
    "GENDER_LABELS",
    "GENDER_TO_INDEX",
    "GENRE_TO_INDEX",
    "MANIFEST_FILENAME",
    "ML1M_URL",
    "MovieLens1MData",
    "MovieRecord",
    "RatingRecord",
    "SPLIT_LABELS",
    "UNKNOWN_GENDER_INDEX",
    "UserHistory",
    "UserRecord",
    "download_ml1m",
    "gender_one_hot",
    "iter_movies",
    "iter_ratings",
    "iter_users",
    "load_processed_data",
    "normalize_gender",
    "preprocess_ml1m",
    "resolve_raw_directory",
    "safe_extract_zip",
    "split_title_year",
    "stratified_user_split",
]
