# DEE-ColdRec

**Equal-Experience Regularization for Cold-Start Recommendation**

DEE-ColdRec recommends warm-catalog MovieLens 1M movies to a completely
unseen user from gender and zero or more 1–5 ratings. There is no learned
user-ID embedding. The repository contains two deliberately separated studies:

1. the original deterministic, TaNP-inspired conditional set encoder and its
   naturally trained baseline/DEE pair; and
2. the current **Wide-and-Deep explicit gender×genre controlled-bias stress
   experiment**, designed to make DEE mitigation easy to inspect under a known
   intervention.

In both studies, support `(movie, rating)` pairs are encoded and mean-pooled
into a user state, then decoded against candidate movies.

The fair model adds the differentiable equal-experience (DEE) regularizer from
[Cho, Choi & Suh (2022)](https://arxiv.org/pdf/2210.05936). See
[`FINAL_REPORT.md`](FINAL_REPORT.md) for equations, protocol details, complete
results, limitations, and the model-selection rationale.

## Current comparison: explicit gender×genre stress experiment

> **Important:** the stress baseline intentionally injects a prescribed
> gender×genre separation for a controlled fairness experiment. It is not an
> estimate of real male or female movie preferences, must not be interpreted as
> evidence about gender, and must not be used for ordinary recommendations or
> decisions about people.

The `explicit_gender_genre` architecture keeps the few-shot deep taste encoder
but makes gender enter only through a centered, interpretable 18-genre wide
term. Male and female receive opposite signs, while unknown/prefer-not-to-answer
is exactly neutral. The four arms of the current comparison are:

| Arm | Checkpoint | Training role |
|:---|:---|:---|
| Natural explicit baseline | `explicit_natural_best.pt` | Rating MSE only; no DEE and no injected stress objective. |
| Controlled-bias stress baseline | `stress_baseline_best.pt` | Warm-started from the natural model and deliberately trained with the labeled gender×genre margin objective. |
| MSE recovery control | `stress_control_best.pt` | Warm-started from the same stress checkpoint and continued with ordinary MSE only, isolating continued-training effects from DEE. |
| DEE mitigation model | `stress_fair_best.pt` | Warm-started from the same stress checkpoint with stress injection off and DEE on. |

The original `baseline_best.pt`/`fair_best.pt` pair remains available and is
reported later in this README. It is a separate natural-data experiment and
must not be conflated with the deliberately biased stress baseline. Under the
same full-catalog support-5 gender-toggle audit, the original baseline has mean
absolute score change `0.00495`, Top-10 overlap `0.9980`, and paper-group DID
approximately zero (`-0.000003`). Its natural gender effect is therefore tiny;
the large stress behavior below is an explicit intervention.

### Test results at support size 5

All four models use the same 604 user-disjoint test users.

| Arm | RMSE | NDCG@10 | DEE |
|:---|---:|---:|---:|
| Natural explicit baseline | 0.977001 | 0.857917 | 0.036755 |
| Controlled-bias stress baseline | 0.989962 | 0.854992 | 0.410436 |
| MSE recovery control | 0.974442 | 0.859339 | 0.216369 |
| DEE mitigation model | 0.992520 | 0.858909 | 0.030866 |

The same-support counterfactual audit holds every support movie, rating, and
timestamp fixed and changes only the M/F input. At support size 5:

| Audit metric | Stress baseline | DEE mitigation |
|:---|---:|---:|
| Mean Top-10 overlap fraction | 0.2301 | 0.7897 |
| Gender×item-group difference-in-differences | 0.676663 | 0.088134 |
| Top-10 genre-exposure L1 distance | 2.3603 | 0.6046 |
| DEE, `tau=3.0` | 0.410695 | 0.030880 |
| DEE, `tau=3.5` | 0.401048 | 0.058141 |
| DEE, `tau=4.0` | 0.206679 | 0.051308 |
| Rank-DEE | 0.013337 | 0.013981 |

DEE substantially reduces predicted-preference and counterfactual exposure
differences while preserving NDCG in this controlled scenario. It does **not**
improve every fairness measurement: rank-DEE rises slightly from `0.013337` to
`0.013981`. The MSE recovery control also removes part of the injected gap, so
the stress→fair change must be interpreted together with that control rather
than attributed wholly to DEE.

## Reproduce

Clone the repository and run the commands from its root directory:

```bash
git clone https://github.com/thdwpfl/dee-coldrec.git
cd dee-coldrec
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

The environment needs PyTorch 2.2+ built for its CUDA runtime. The run reported
here used Python 3.12.3, PyTorch `2.10.0a0+b4e4ee81d3.nv25.12`, and one NVIDIA
B200. Training commands explicitly expose one GPU and the code rejects a CUDA
run if PyTorch sees anything other than one device.

Download and deterministically preprocess MovieLens 1M:

```bash
.venv/bin/python scripts/download_data.py --seed 20260729
```

Run the unit and integration tests:

```bash
.venv/bin/python -m pytest -q
```

The current verified suite reports `29 passed`.

Run small end-to-end baseline and fair smoke training:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py \
  --config configs/smoke.yaml --output-dir artifacts/smoke/baseline

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py \
  --config configs/smoke_fair.yaml --output-dir artifacts/smoke/fair \
  --initial-checkpoint artifacts/smoke/baseline/best.pt
```

Run the bounded full-data search (three baseline and four fair trials):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/tune.py \
  --baseline-config configs/baseline.yaml \
  --fair-config configs/fair.yaml \
  --search-config configs/search.yaml \
  --output-root artifacts
```

### Reproduce the explicit controlled-bias comparison

Run the small explicit-architecture natural, stress, and DEE smoke sequence:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py \
  --config configs/explicit_smoke_natural.yaml \
  --output-dir artifacts/smoke/explicit_natural

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py \
  --config configs/explicit_smoke_stress.yaml \
  --output-dir artifacts/smoke/explicit_stress \
  --initial-checkpoint artifacts/smoke/explicit_natural/best.pt

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py \
  --config configs/explicit_smoke_fair.yaml \
  --output-dir artifacts/smoke/explicit_fair \
  --initial-checkpoint artifacts/smoke/explicit_stress/best.pt
```

Run the bounded validation-only search. This trains the natural baseline,
controlled stress trials, MSE recovery control, and DEE mitigation trials, then
copies the selected checkpoints without consulting the test split:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_gender_stress_experiment.py \
  --natural-config configs/explicit_natural.yaml \
  --stress-config configs/explicit_stress.yaml \
  --fair-config configs/explicit_fair.yaml \
  --search-config configs/gender_stress_search.yaml \
  --output-root artifacts/gender_stress
```

Evaluate all four selected checkpoints on the held-out test users:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/explicit_natural_best.pt \
  --data data/processed --output artifacts/metrics/explicit_natural_test.json \
  --support-sizes 1 3 5 10

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/stress_baseline_best.pt \
  --data data/processed --output artifacts/metrics/stress_baseline_test.json \
  --support-sizes 1 3 5 10

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/stress_control_best.pt \
  --data data/processed --output artifacts/metrics/stress_control_test.json \
  --support-sizes 1 3 5 10

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/stress_fair_best.pt \
  --data data/processed --output artifacts/metrics/stress_fair_test.json \
  --support-sizes 1 3 5 10
```

Audit the same four checkpoints by flipping only gender while holding each
user's support set fixed. The audit reports Top-10 overlap, difference-in-
differences, genre exposure, and DEE at thresholds 3.0, 3.5, and 4.0:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/audit_gender_effect.py \
  --checkpoint artifacts/checkpoints/explicit_natural_best.pt \
  --data data/processed --output artifacts/metrics/explicit_natural_gender_audit.json \
  --support-sizes 1 3 5 10 --thresholds 3 3.5 4

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/audit_gender_effect.py \
  --checkpoint artifacts/checkpoints/stress_baseline_best.pt \
  --data data/processed --output artifacts/metrics/stress_baseline_gender_audit.json \
  --support-sizes 1 3 5 10 --thresholds 3 3.5 4

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/audit_gender_effect.py \
  --checkpoint artifacts/checkpoints/stress_control_best.pt \
  --data data/processed --output artifacts/metrics/stress_control_gender_audit.json \
  --support-sizes 1 3 5 10 --thresholds 3 3.5 4

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/audit_gender_effect.py \
  --checkpoint artifacts/checkpoints/stress_fair_best.pt \
  --data data/processed --output artifacts/metrics/stress_fair_gender_audit.json \
  --support-sizes 1 3 5 10 --thresholds 3 3.5 4
```

Evaluate the original selected checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/baseline_best.pt \
  --data data/processed --output artifacts/metrics/baseline_test.json \
  --support-sizes 1 3 5 10

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/fair_best.pt \
  --data data/processed --output artifacts/metrics/fair_test.json \
  --support-sizes 1 3 5 10
```

Run an inference example:

```bash
.venv/bin/python scripts/recommend.py \
  --checkpoint artifacts/checkpoints/fair_best.pt \
  --gender female --rating 1:5 --rating 260:5 --rating 1196:5 \
  --rating 1721:2 --top-k 10
```

## Website

Launch the verified DEE-regularized Streamlit site:

```bash
.venv/bin/streamlit run app.py
```

Launch the separate verified baseline site, which uses the trained checkpoint
with `dee_coefficient: 0.0`:

```bash
.venv/bin/streamlit run app_baseline.py
```

To compare both in separate browser tabs at the same time, run one command in
each of two terminals:

```bash
.venv/bin/streamlit run app.py --server.port 8501
.venv/bin/streamlit run app_baseline.py --server.port 8502
```

Launch the controlled-bias stress baseline and its DEE mitigation model in two
additional tabs:

```bash
.venv/bin/streamlit run app_stress_baseline.py --server.port 8503
.venv/bin/streamlit run app_stress_fair.py --server.port 8504
```

Both stress sites display a persistent Korean warning directly under the model
badge on the rating and recommendation screens. The warning states that the
behavior was intentionally induced for a controlled experiment and does not
represent actual gender preferences.

The app uses a Korean, two-step flow. First, choose gender (including
unknown/prefer-not-to-answer), search the popularity-ordered local catalog, and
give explicit 1–5 star ratings to at least three watched movies. Movie IDs are
never shown; every choice and rating card shows title, year, and Korean genre
labels. MovieLens suffix articles are displayed naturally (`Title, The` becomes
`The Title`) without changing the canonical dataset. After submission, a
separate results screen presents 10
title/year/genre/score cards,
with controls to edit the ratings or reset the session. Rated movies are
excluded, the genre-distribution diagnostic is optional, inputs stay only in
Streamlit session state, and no external key is needed. The reusable inference
API still supports zero or very few ratings, while the website deliberately
requires three for a clearer and more useful onboarding experience. The results
screen explicitly confirms whether male, female, or no gender information was
applied to the model. Run the
two-screen interactive headless smoke tests with:

```bash
.venv/bin/python scripts/smoke_test_app.py \
  --app app.py \
  --expected-checkpoint artifacts/checkpoints/fair_best.pt \
  --expected-model-label 'DEE 공정성 적용 모델'

.venv/bin/python scripts/smoke_test_app.py \
  --app app_baseline.py \
  --expected-checkpoint artifacts/checkpoints/baseline_best.pt \
  --expected-model-label 'Baseline 추천 모델 · DEE 미적용'

.venv/bin/python scripts/smoke_test_app.py \
  --app app_stress_baseline.py \
  --expected-checkpoint artifacts/checkpoints/stress_baseline_best.pt \
  --expected-model-label '통제 편향 Stress Baseline · DEE 미적용' \
  --expected-notice '실제 성별에 따른 영화 취향을 대표하지 않으며'

.venv/bin/python scripts/smoke_test_app.py \
  --app app_stress_fair.py \
  --expected-checkpoint artifacts/checkpoints/stress_fair_best.pt \
  --expected-model-label '통제 편향 완화 모델 · DEE 적용' \
  --expected-notice '실제 성별에 따른 영화 취향을 대표하지 않으며'
```

To use alternate artifacts, set `COLDSTART_CHECKPOINT` and `COLDSTART_DATA`.
The baseline was not retrained for the second site: its checkpoint already
contains an epoch-7 full-data model trained with one visible GPU, and its
embedded configuration plus logs verify that the DEE training term was zero.

## Original TaNP-inspired baseline/fair test results

Ranking candidates are each unseen test user's timestamp-held-out rated query
movies; full-catalog top-10 lists are used for recommendation exposure. A
rating of at least 4 is relevant for ranking. DEE is audited densely at common
`tau=3` and `h=0.5` over every unambiguous paper-group movie.

| Support | Model | RMSE | Recall@10 | NDCG@10 | HitRate@10 | DEE |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | baseline | 0.9905 | 0.2072 | 0.8578 | 1.0000 | 0.01450 |
| 1 | fair | 1.0055 | 0.2070 | 0.8569 | 1.0000 | 0.00547 |
| 3 | baseline | 0.9782 | 0.2168 | 0.8587 | 1.0000 | 0.01421 |
| 3 | fair | 0.9991 | 0.2168 | 0.8590 | 1.0000 | 0.00528 |
| 5 | baseline | 0.9747 | 0.2292 | 0.8593 | 1.0000 | 0.01569 |
| 5 | fair | 0.9956 | 0.2278 | 0.8573 | 1.0000 | 0.00643 |
| 10 | baseline | 0.9686 | 0.2701 | 0.8583 | 1.0000 | 0.01805 |
| 10 | fair | 0.9902 | 0.2688 | 0.8565 | 1.0000 | 0.00535 |

Across these support sizes, mean DEE decreased 63.9%, mean RMSE increased 2.0%,
and mean NDCG@10 decreased 0.13%.

## Main artifacts

- `artifacts/checkpoints/baseline_best.pt`
- `artifacts/checkpoints/fair_best.pt`
- `artifacts/tuning_summary.json`
- `artifacts/metrics/baseline_test.json`
- `artifacts/metrics/fair_test.json`
- `artifacts/example_recommendations.json`
- `configs/selected_baseline.yaml`
- `configs/selected_fair.yaml`
- `artifacts/checkpoints/explicit_natural_best.pt`
- `artifacts/checkpoints/stress_baseline_best.pt`
- `artifacts/checkpoints/stress_control_best.pt`
- `artifacts/checkpoints/stress_fair_best.pt`
- `artifacts/metrics/explicit_natural_gender_audit.json`
- `artifacts/metrics/stress_baseline_gender_audit.json`
- `artifacts/metrics/stress_control_gender_audit.json`
- `artifacts/metrics/stress_fair_gender_audit.json`
- `configs/selected_explicit_natural.yaml`
- `configs/selected_stress_baseline.yaml`
- `configs/selected_stress_control.yaml`
- `configs/selected_stress_fair.yaml`
- `data/processed/split_manifest.json`
