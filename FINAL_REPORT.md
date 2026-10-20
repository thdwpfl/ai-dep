# Final report: cold-start movie recommendation with DEE

## Completion and verification

The complete system is implemented, trained, tuned, evaluated, and tested in
`/NHNHOME/WORKSPACE/26msit001_A/optiml/aidep`.

- Dataset: MovieLens 1M, downloaded from GroupLens
- Full data: 6,040 users, 3,883 movies, 1,000,209 ratings
- Hardware used: exactly one visible NVIDIA B200 (`cuda:0`)
- Final test users: 604 users unseen during all training and selection
- Tests: 29/29 passed
- Headless Streamlit interactions: the original fair/baseline sites and the
  new controlled-stress baseline/DEE sites all passed the three-rating gate,
  advanced to a separate results screen, and rendered 10 recommendations from
  their exact intended checkpoints
- Baseline and DEE checkpoints: present and inference-tested
- Final inference examples: present in `artifacts/example_recommendations.json`
  and `artifacts/example_recommendations_baseline.json`, plus gender-toggle
  stress examples under `artifacts/examples/`

No result below is fabricated. Metrics come from the saved JSON artifacts.

## Controlled-bias demonstrator added after the initial system

The user requested a second model whose baseline has an intentionally large,
visible gender dependence so that DEE can be demonstrated as a mitigation.
That request is implemented as a separate, explicitly labeled scientific
stress test. It does **not** relabel ordinary MovieLens behavior as social
bias, and it does **not** claim that real male and female users should prefer
different genres.

The final comparison has four arms:

1. **Natural explicit baseline**: new architecture, original data, MSE only,
   random initialization. This measures the naturally learned gender effect.
2. **Controlled-bias stress baseline**: warm-start the natural model and add a
   disclosed gender-by-genre margin objective. DEE is zero in this arm.
3. **MSE recovery control**: warm-start the selected stress checkpoint, remove
   the stress term, and continue with MSE only. This prevents ordinary recovery
   from being incorrectly attributed to DEE.
4. **DEE mitigation**: warm-start the exact same selected stress checkpoint,
   remove the stress term, and optimize MSE plus paper-faithful DEE.

The stress and DEE websites carry a prominent warning on both screens that
this is an intentionally biased controlled experiment, not evidence of actual
gender-specific taste and not a model for personal decision making.

For context, the original natural `baseline_best.pt` was also rerun through
the same full-catalog test audit. At support 5, flipping only gender changes a
score by `0.004947` on average, retains `99.80%` of Top-10 items, keeps Top-1
for `99.83%` of users, and gives paper-group DID `-0.000003`. The original
baseline is not substantially gender-sensitive; the new stress baseline is
deliberately constructed to make the mitigation scenario measurable.

### New model: explicit gender-by-genre Wide-and-Deep cold start

The alternative architecture retains the reliable few-shot DeepSets taste
tower but removes gender from that deep tower. It adds an interpretable wide
cross head over the 18 MovieLens genres. It is therefore a Wide-and-Deep
cold-start model rather than the original implicit-gender set encoder. It still
has no learned user-ID table: an unseen user's deep state comes only from the
chronological support `(movie,rating)` pairs.

For movie `i`, let `x_i` be its multi-hot genre vector, `d` the learned
18-dimensional genre coefficient, and `z(u,i)` the deep taste logit. The final
logit and rating are:

```text
s(M) = -0.5,  s(F) = +0.5,  s(unknown) = 0
wide(g,i) = alpha * s(g) * <x_i,d> / max(1, ||x_i||_1)
r_hat(u,i,g) = 3 + 2*tanh((z(u,i) + wide(g,i)) / 2)
```

The selected `alpha` is 1.0. Unknown/prefer-not-to-answer is exactly neutral;
M/F move symmetrically in opposite directions. `with_gender()` changes only
this sign while preserving the same support-derived taste state, which makes a
same-history counterfactual audit well defined. The model has 362,878 trainable
parameters. Directionality, unknown neutrality, checkpoint round-trip, finite
gradients, and service behavior are covered by tests.

The disclosed stress objective uses the paper's two non-overlapping item
groups. Let group 0 be Action/Crime/Film-Noir/War and group 1 be
Children's/Fantasy/Musical/Romance:

```text
d0 = mean(r_hat(M,G0) - r_hat(F,G0))
d1 = mean(r_hat(F,G1) - r_hat(M,G1))
L_stress = relu(margin - d0) + relu(margin - d1)
L_baseline_stress = MSE + beta * L_stress
L_fair = MSE + lambda_DEE * Huber_DEE
```

The selected stress arm uses `margin=0.6`, `beta=1.0`, and
`dee_coefficient=0`. The selected fair arm starts from that stress checkpoint,
sets `stress_coefficient=0`, and uses `lambda_DEE=100`, KDE bandwidth `h=0.5`,
threshold `tau=3`, and Huber delta `0.01`. Thus the fair comparison changes the
objective, not the architecture, source checkpoint, catalog, or user split.

### Bounded search and validation-only selection

All 11 full-data trials completed on one visible NVIDIA B200; none failed and
all values in the logs are finite. Natural used random initialization; every
stress trial started from natural; control and every fair trial started from
the selected stress checkpoint. Selection used only 604 validation users at
five-shot support. The 604 test users remained untouched until final selection.

| Trial | Best epoch | RMSE | NDCG@10 | Recall@10 | DEE | CF interaction | Status |
|:---|---:|---:|---:|---:|---:|---:|:---|
| Natural | 3 | 0.981456 | 0.832990 | 0.352020 | 0.050536 | 0.067709 | reference |
| Stress m=.2 beta=.3 | 6 | 0.978596 | 0.832042 | 0.352037 | 0.225014 | 0.433689 | gate passed |
| Stress m=.4 beta=.3 | 6 | 0.994184 | 0.830480 | 0.352777 | 0.370595 | 0.643922 | gate passed |
| Stress m=.4 beta=1 | 6 | 0.985759 | 0.826529 | 0.351586 | 0.399204 | 0.670448 | gate passed |
| Stress m=.6 beta=.3 | 6 | 0.982013 | 0.831003 | 0.352220 | 0.334758 | 0.628585 | gate passed |
| **Stress m=.6 beta=1** | **6** | **0.994238** | **0.825498** | **0.350059** | **0.410745** | **0.676819** | **selected** |
| MSE recovery control | 4 | 0.976109 | 0.835081 | 0.352640 | 0.215315 | 0.451029 | control |
| Fair lambda=25 h=.50 | 8 | 0.975306 | 0.837321 | 0.353698 | 0.059033 | 0.143490 | gate passed |
| Fair lambda=100 h=.25 | 8 | 1.001201 | 0.837103 | 0.354870 | 0.055119 | 0.202189 | gate passed |
| **Fair lambda=100 h=.50** | **8** | **0.992320** | **0.838629** | **0.354495** | **0.036561** | **0.088176** | **selected** |
| Fair lambda=250 h=.25 | 8 | 1.002802 | 0.833196 | 0.351504 | 0.053618 | 0.205866 | gate passed |

The predeclared stress gate required at least 0.3 counterfactual interaction,
at most 10% RMSE degradation, and at least 90% NDCG retention. The fair gate
required at least 50% reductions in both DEE and interaction, at most 5% RMSE
degradation, and at least 95% NDCG retention. All bounded candidates passed
their respective gates. The selected stress trial maximized the interaction;
the selected fair trial minimized the normalized DEE-plus-interaction score.

### Final independent test results for the four-arm experiment

These are all 604 user-disjoint test users. HitRate@10 is 1.0 in every row and
is omitted from the table because it is saturated. Complete MAE, per-gender
metrics, probabilities, exposures, and HitRate are in the linked JSON files.

| Support | Model | RMSE | MAE | Recall@10 | NDCG@10 | DEE | Ranking-DEE |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 1 | Natural | 0.988878 | 0.788693 | 0.208318 | 0.856796 | 0.032978 | 0.014920 |
| 1 | Stress | 0.998520 | 0.795872 | 0.206976 | 0.853738 | 0.415266 | 0.013965 |
| 1 | Control | 0.990817 | 0.790936 | 0.207135 | 0.857247 | 0.219418 | 0.011309 |
| 1 | Fair | 0.999684 | 0.801276 | 0.206173 | 0.857722 | 0.030757 | 0.014391 |
| 3 | Natural | 0.977734 | 0.779873 | 0.218211 | 0.858453 | 0.037154 | 0.014499 |
| 3 | Stress | 0.992720 | 0.790593 | 0.216038 | 0.854059 | 0.411530 | 0.013614 |
| 3 | Control | 0.979002 | 0.780677 | 0.218454 | 0.860183 | 0.216764 | 0.011393 |
| 3 | Fair | 0.995730 | 0.797764 | 0.216879 | 0.859943 | 0.030713 | 0.014055 |
| 5 | Natural | 0.977001 | 0.778482 | 0.228874 | 0.857917 | 0.036755 | 0.014341 |
| 5 | Stress | 0.989962 | 0.787648 | 0.227784 | 0.854992 | 0.410436 | 0.013337 |
| 5 | Control | 0.974442 | 0.775795 | 0.229764 | 0.859339 | 0.216369 | 0.011198 |
| 5 | Fair | 0.992520 | 0.795207 | 0.228373 | 0.858909 | 0.030866 | 0.013981 |
| 10 | Natural | 0.973594 | 0.774717 | 0.270727 | 0.855242 | 0.035150 | 0.014260 |
| 10 | Stress | 0.985840 | 0.784082 | 0.266978 | 0.850994 | 0.408766 | 0.013224 |
| 10 | Control | 0.966446 | 0.767934 | 0.270009 | 0.857829 | 0.213355 | 0.011182 |
| 10 | Fair | 0.986641 | 0.789538 | 0.268772 | 0.855963 | 0.030227 | 0.014062 |

Across support sizes, stress to fair changed mean RMSE by +0.19%, improved
mean NDCG@10 by 0.55%, and reduced mean score-DEE by 92.55%. At support 5,
RMSE changed by +0.26%, NDCG improved by 0.46%, and DEE fell by 92.48%.
However, mean Ranking-DEE changed from 0.013535 to 0.014122 (+4.34%): this
experiment does **not** demonstrate improvement on the paper's top-K extension.

The MSE-only recovery control matters. Relative to stress, it reduced mean DEE
by 47.39% and validation interaction by 33.36% without DEE. The fair model
reduced mean DEE by 92.55% and interaction by 86.97%. This shows a substantial
additional association with DEE training, but it is not a formal causal
decomposition because the control and fair runs use different seeds and select
epochs under different objectives.

Five-shot accuracy by actual test gender:

| Model | Gender | Users | RMSE | MAE | Recall@10 | NDCG@10 |
|:---|:---|---:|---:|---:|---:|---:|
| Natural | Male | 433 | 0.956013 | 0.764185 | 0.223449 | 0.864447 |
| Natural | Female | 171 | 1.043617 | 0.825918 | 0.242612 | 0.841384 |
| Stress | Male | 433 | 0.969054 | 0.772700 | 0.225546 | 0.867758 |
| Stress | Female | 171 | 1.056373 | 0.837246 | 0.233449 | 0.822669 |
| Control | Male | 433 | 0.953966 | 0.760944 | 0.225922 | 0.867534 |
| Control | Female | 171 | 1.039495 | 0.825065 | 0.239491 | 0.838589 |
| Fair | Male | 433 | 0.975468 | 0.783207 | 0.223662 | 0.866425 |
| Fair | Female | 171 | 1.047109 | 0.835021 | 0.240303 | 0.839875 |

At five shots, the mean absolute actual-gender exposure gap over all 18 genres
was 0.00886 natural, 0.13084 stress, 0.11382 control, and 0.03417 fair. Stress
to fair therefore reduced this diagnostic by 73.9%, but fair remains above the
natural model. The largest residual fair gap was Drama (male 0.3921, female
0.5860, absolute 0.1938), which is outside the two paper genre groups.

### Same-history counterfactual test audit

`scripts/audit_gender_effect.py` holds every support movie, rating, timestamp,
and catalog candidate fixed, changes only M/F, excludes rated support, and
scores the entire 3,883-movie catalog. It records checkpoint SHA-256, dataset
fingerprint, exact GPU visibility, score distributions, Top-10 set/order
changes, genre exposure, paper-group difference-in-differences, and DEE at
three thresholds.

Five-shot results:

| Model | Mean absolute score delta | Top-10 overlap | Top-1 same | Paper-group DID | 18-genre exposure L1 |
|:---|---:|---:|---:|---:|---:|
| Natural | 0.021522 | 0.9523 | 0.9536 | 0.067638 | 0.1613 |
| Stress | 0.261061 | 0.2301 | 0.0000 | 0.676663 | 2.3603 |
| Control | 0.162465 | 0.2737 | 0.0000 | 0.451232 | 2.0598 |
| Fair | 0.033517 | 0.7897 | 0.9917 | 0.088134 | 0.6046 |

Thus stress produces a large and visible gender-toggle effect. DEE increases
Top-10 overlap from 23.0% to 79.0%, reduces the paper-group interaction by
87.0%, and reduces counterfactual 18-genre exposure L1 by 74.4%. Exact Top-10
set equality remains zero for fair because at least one lower-ranked item still
changes, even though Top-1 is identical for 99.17% of users.

Threshold sensitivity at support 5:

| Threshold | Natural DEE | Stress DEE | Control DEE | Fair DEE | Stress-to-fair reduction |
|---:|---:|---:|---:|---:|---:|
| 3.0 | 0.036573 | 0.410695 | 0.216453 | 0.030880 | 92.48% |
| 3.5 | 0.061822 | 0.401048 | 0.217139 | 0.058141 | 85.50% |
| 4.0 | 0.075927 | 0.206679 | 0.146241 | 0.051308 | 75.18% |

Fair's mean audited score rose from stress 3.3069 to 3.6110 and its global
preference probability at `tau=3` rose from 0.6732 to 0.8491. This confirms a
distribution shift and means calibration remains a concern. Nevertheless,
DEE also falls strongly at thresholds 3.5 and 4.0, so the observed reduction
is not explained solely by saturation at threshold 3.0.

### Selected artifacts, hashes, sites, and examples

- Natural: `artifacts/checkpoints/explicit_natural_best.pt`, epoch 3,
  SHA-256 `ec6ce957aeabe2923aec02dfb0516245baf57d58982cb93400273f4132ac51c7`
- Stress baseline: `artifacts/checkpoints/stress_baseline_best.pt`, epoch 6,
  SHA-256 `462e49d736f2f2247a33812e39fbe33f4ef100ed1d04195bd848d990a4e05fd5`
- MSE control: `artifacts/checkpoints/stress_control_best.pt`, epoch 4,
  SHA-256 `50a759761134dec733a103498a056e090d185f30e69fba723e20946e9abe5092`
- DEE mitigation: `artifacts/checkpoints/stress_fair_best.pt`, epoch 8,
  SHA-256 `b6f7d93fac3f1894e9d15b0324aeab9234d6f933037732afa9f4a08294f4fa08`

Exact selected YAMLs are `configs/selected_explicit_natural.yaml`,
`configs/selected_stress_baseline.yaml`,
`configs/selected_stress_control.yaml`, and
`configs/selected_stress_fair.yaml`. All configs, epoch JSONL logs, summaries,
and trial checkpoints are under `artifacts/gender_stress/`. The selection
summary is `artifacts/gender_stress/tuning_summary.json`; full test metrics and
counterfactual audits are under `artifacts/metrics/`.
The original-baseline comparison audit is
`artifacts/metrics/baseline_gender_effect_audit.json`.

Launch the deliberately biased comparison baseline and its DEE mitigation in
separate terminals:

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/optiml/aidep
.venv/bin/streamlit run app_stress_baseline.py --server.port 8503
.venv/bin/streamlit run app_stress_fair.py --server.port 8504
```

Both sites use the same user-friendly two-screen flow as the original site:
no visible IDs, direct star controls, minimum three ratings, title/year/Korean
genres, session-only data, rated-item exclusion, reset, and optional genre
diagnostics. Clicking the recommendation button executes the selected model;
gender changes the explicit wide head, while unknown uses a zero wide effect.

For the fixed profile `1:5, 260:5, 1196:5, 1721:2`, the stress baseline M/F
Top-10 lists overlap by 2/10, while the fair M/F lists overlap by 8/10. The
stress male list starts with *Rear Window*, *The Sixth Sense*, and *The Usual
Suspects*; the stress female list starts with *The Wrong Trousers*, *The
Shawshank Redemption*, and *Wallace & Gromit: The Best of Aardman Animation*.
All four complete lists are stored in `artifacts/examples/`.

The defensible conclusion is: DEE strongly mitigated a deliberately induced,
controlled gender-by-genre **score dependence** while retaining test quality.
It did not prove that real-world gender bias was solved, did not improve
Ranking-DEE, and did not fully restore natural-model genre exposure.

## Original model candidates and first production choice

| Candidate | Assessment |
|:---|:---|
| MeLU | Conceptually appropriate MAML cold-start model with public code, but its official implementation targets Python 3.6/PyTorch 1.1, performs serial inner-loop state mutation, and makes joint cross-user DEE training and zero-shot serving unnecessarily fragile. |
| Full TaNP | Public and designed for few-shot user tasks, but the official Python 3.6/PyTorch 1.1 implementation adds latent sampling, clustering, KL, and FiLM complexity that is not needed for this deliverable. |
| MetaHIN/MAMO | Supports cold-start, but requires richer heterogeneous metadata or expensive serial meta-learning that does not match the deployable input contract. |
| DropoutNet | Strong content cold-start precedent, but is a two-stage WMF/content-dropout method and is not naturally conditioned on a small explicit rating set. |
| **Selected: deterministic TaNP-inspired conditional set encoder** | Directly represents an arbitrary-size support set, is fast and batchable, supports zero or one rating, leaves the global checkpoint unchanged at inference, and keeps joint gender×genre DEE in a normal autograd graph. |

The selected implementation follows the Conditional Neural Process/DeepSets
encode–aggregate–decode pattern that underlies TaNP's amortized new-user
inference. It is deliberately described as **TaNP-inspired**, not as an exact
TaNP reproduction.

For each support pair, the model combines a learned warm-item embedding,
18-genre features, normalized release year, and normalized rating. A shared MLP
encodes each pair; masked mean pooling makes the support representation
permutation invariant. The pooled context, `log(1+k)`, and a three-way gender
embedding (male/female/unknown) form the new-user state. The decoder scores
catalog items in `[1,5]`. There is no user-ID input or user embedding table.
Learned movie-ID embeddings are appropriate because this is user cold-start,
not item cold-start.

The implementation has 363,908 trainable parameters. Gender dropout sends 10%
of training profiles through the unknown token so that prefer-not-to-answer
inference is defined.

Primary model references:

- [MeLU paper and public implementation](https://github.com/hoyeoplee/MeLU)
- [TaNP paper](https://arxiv.org/abs/2103.06137) and
  [public implementation](https://github.com/IIEdm/TaNP)
- [Conditional Neural Processes](https://arxiv.org/abs/1807.01613)
- [Deep Sets](https://arxiv.org/abs/1703.06114)

## Data and genuine cold-start split

`scripts/download_data.py` downloads and safely extracts MovieLens 1M, parses
the Latin-1 `::` files, strips title years, constructs all 18 genre tags, sorts
every user history by timestamp, and emits a compressed NPZ plus a manifest.
The processed artifact SHA-256 is
`51ef38f20b07e20b64058dffc96e3d31fbf54f72ace189a38117190cb791a759`.

The split is deterministic (`seed=20260729`), user-disjoint, and stratified by
gender:

| Split | Users | Ratings | Male users | Female users |
|:---|---:|---:|---:|---:|
| Train | 4,832 | 800,159 | 3,465 | 1,367 |
| Validation | 604 | 97,986 | 433 | 171 |
| Test | 604 | 102,064 | 433 | 171 |

There is no user overlap. Within each validation/test user, the first
`k ∈ {1,3,5,10}` timestamp-ordered ratings form support and every later rating
forms query. Train episodes randomly choose `k ∈ {0,1,3,5,10}` and sample up
to 48 later query ratings.

RMSE/MAE use all held-out query ratings. Recall@10, NDCG@10, and HitRate@10
rank the held-out rated query set, with rating >=4 considered relevant. This
avoids falsely treating the many unobserved MovieLens entries as dislikes.
Full-catalog predictions excluding support movies are used for recommendation
and genre-exposure diagnostics.

## Paper-faithful DEE implementation

The implementation follows Cho, Choi & Suh, *Equal Experience in Recommender
Systems*, [arXiv:2210.05936](https://arxiv.org/pdf/2210.05936):
Definition 3.2/Eq. 12, KDE Eqs. 14–15, and the Huber-gradient implementation in
Eqs. 20–21.

For predicted rating `s`, threshold `tau`, and Gaussian KDE bandwidth `h`, the
differentiable preference contribution is

```text
q(s) = Phi((s - tau) / h).
```

For joint user group `a` and item group `b`,

```text
DEE = sum_(a,b) | mean(q) - mean(q | Z_user=a, Z_item=b) |.
```

This is an **unweighted sum**, not a cell average and not gender-only
demographic parity. Independence is enforced against the joint
`(gender,item-group)` variable. Evaluation reports the raw absolute sum. The
training loss replaces each absolute gap with the paper's unscaled Huber
surrogate:

```text
H_delta(d) = 0.5*d^2                         if |d| <= delta
             delta*(|d| - 0.5*delta)        otherwise.
```

`tests/test_fairness.py` verifies:

1. group-independent score distributions give DEE approximately zero;
2. deliberately biased joint cells give positive DEE, including an arrangement
   that marginal-only checks would miss;
3. the Huber/KDE loss has finite, nonzero gradients;
4. multi-label expansion and empty-membership behavior.

MovieLens uses `tau=3` as in the paper. Training tuned `h`; all validation/test
comparisons use common `h=0.5` and `delta=0.01`.

The paper's MovieLens experiment defines two item groups:

- male-preferred: Action, Crime, Film-Noir, War;
- female-preferred: Children's, Fantasy, Musical, Romance.

The source paper does not specify how multi-genre movies tagged by both sets or
neither set are assigned. This implementation uses a transparent categorical
rule: 698 movies tagged only by the first set and 714 tagged only by the second
enter DEE; the 2,471 ambiguous/neither movies remain in accuracy training and
recommendation but are excluded from DEE. This makes `Z_item` categorical as
required by Eq. 12. All 18 multi-label genres are still reported in the
exposure diagnostic.

Training estimates the dense user×item preference matrix with a balanced
96-item stochastic catalog sample per batch. Validation and test DEE are exact
over every one of the 1,412 unambiguous catalog movies for every audited user.
The optimization is `mean query MSE + coefficient * Huber-DEE`; the larger
coefficient is needed because mean MSE and the paper's unscaled
`delta=0.01` Huber term have different scales.

## Smoke tests, training, and bounded tuning

Both one-epoch smoke runs completed on 128 training users and 48 validation
users. The smoke baseline and fair checkpoints plus logs are under
`artifacts/smoke/`.

The full structured search was bounded to:

- 3 baseline trials varying learning rate, item/hidden/context dimensions, and
  weight decay;
- 4 fair trials varying DEE coefficient and bandwidth;
- 8 epochs per trial;
- all 4,832 training and 604 validation users;
- batch size 64, query sample size 48;
- exactly one GPU for every training run.

All seven final trials completed; no final grid trial failed. The automatic
poor-RMSE and weak-fairness retries were not triggered.

An early tuning audit used a sampled validation catalog and selected a model
that did not improve dense test DEE. That result was rejected. Validation was
corrected to use the exact full paper-group catalog at a common bandwidth and
the bounded search was rerun from scratch. Only the corrected run in
`artifacts/tuning_summary.json` is reported.

Selected validation results:

| Model | Trial | Epoch | RMSE | NDCG@10 | Recall@10 | DEE |
|:---|:---|---:|---:|---:|---:|---:|
| Baseline | `base_64` | 7 | 0.9760 | 0.8306 | 0.3533 | 0.02502 |
| Fair | `dee_100_h025` | 8 | 0.9917 | 0.8372 | 0.3526 | 0.00699 |

The fair model was warm-started from the selected baseline, then trained with
DEE. This preserves the saved baseline and produces a distinct checkpoint.

## Final baseline and fair test results

All values below use all 604 unseen test users and common DEE audit parameters
`tau=3`, `h=0.5`.

| Support | Model | RMSE | MAE | Recall@10 | NDCG@10 | HitRate@10 | DEE | DEE-ranking |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Baseline | 0.9905 | 0.7915 | 0.2072 | 0.8578 | 1.0000 | 0.01450 | 0.02210 |
| 1 | Fair | 1.0055 | 0.8105 | 0.2070 | 0.8569 | 1.0000 | 0.00547 | 0.01686 |
| 3 | Baseline | 0.9782 | 0.7823 | 0.2168 | 0.8587 | 1.0000 | 0.01421 | 0.02207 |
| 3 | Fair | 0.9991 | 0.8046 | 0.2168 | 0.8590 | 1.0000 | 0.00528 | 0.01712 |
| 5 | Baseline | 0.9747 | 0.7789 | 0.2292 | 0.8593 | 1.0000 | 0.01569 | 0.02222 |
| 5 | Fair | 0.9956 | 0.8022 | 0.2278 | 0.8573 | 1.0000 | 0.00643 | 0.01717 |
| 10 | Baseline | 0.9686 | 0.7734 | 0.2701 | 0.8583 | 1.0000 | 0.01805 | 0.02206 |
| 10 | Fair | 0.9902 | 0.7971 | 0.2688 | 0.8565 | 1.0000 | 0.00535 | 0.01739 |

The accuracy–fairness tradeoff is consistent:

| Support | DEE reduction | RMSE change | NDCG@10 change | DEE-ranking reduction |
|---:|---:|---:|---:|---:|
| 1 | 62.3% | +1.5% | -0.10% | 23.7% |
| 3 | 62.8% | +2.1% | +0.03% | 22.5% |
| 5 | 59.1% | +2.1% | -0.23% | 22.7% |
| 10 | 70.4% | +2.2% | -0.21% | 21.2% |

Across support sizes, mean DEE changes from 0.01561 to 0.00563 (-63.9%);
mean RMSE changes from 0.9780 to 0.9976 (+2.0%); and mean NDCG@10 changes
from 0.85853 to 0.85743 (-0.13%). The fair checkpoint therefore substantially
improves the selected fairness measure while preserving most quality.

Five-shot accuracy by gender:

| Model | Gender | Users | RMSE | MAE | Recall@10 | NDCG@10 |
|:---|:---|---:|---:|---:|---:|---:|
| Baseline | Male | 433 | 0.9545 | 0.7643 | 0.2240 | 0.8664 |
| Baseline | Female | 171 | 1.0390 | 0.8272 | 0.2422 | 0.8413 |
| Fair | Male | 433 | 0.9773 | 0.7884 | 0.2225 | 0.8631 |
| Fair | Female | 171 | 1.0539 | 0.8478 | 0.2412 | 0.8426 |

Five-shot top-10 genre exposure by gender (fraction of recommendation
positions carrying each tag; multi-label rows need not sum to one):

| Genre | Baseline M | Baseline F | Fair M | Fair F |
|:---|---:|---:|---:|---:|
| Action | 0.2947 | 0.2965 | 0.3021 | 0.3041 |
| Adventure | 0.1000 | 0.1000 | 0.1097 | 0.1094 |
| Animation | 0.0132 | 0.0135 | 0.2012 | 0.2006 |
| Children's | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Comedy | 0.0254 | 0.0257 | 0.2000 | 0.1994 |
| Crime | 0.1882 | 0.1912 | 0.1935 | 0.1953 |
| Documentary | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Drama | 0.4995 | 0.5035 | 0.3952 | 0.3965 |
| Fantasy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Film-Noir | 0.1000 | 0.1000 | 0.0942 | 0.0942 |
| Horror | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Musical | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Mystery | 0.0998 | 0.0994 | 0.0972 | 0.0982 |
| Romance | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Sci-Fi | 0.0945 | 0.0889 | 0.0000 | 0.0000 |
| Thriller | 0.1928 | 0.1942 | 0.2995 | 0.2994 |
| War | 0.2880 | 0.2842 | 0.1023 | 0.1018 |
| Western | 0.0122 | 0.0123 | 0.0000 | 0.0000 |

The complete per-support, per-gender, cell-probability, DEE-ranking, and
18-genre diagnostics are saved in the two final metric JSON files.

## Selected configuration and checkpoints

Common architecture:

- item embedding dimension: 64
- hidden dimension: 128
- context dimension: 96
- gender embedding dimension: 8
- dropout / gender dropout: 0.10 / 0.10
- batch size: 64
- learning rate: 0.001
- weight decay: 1e-5
- epochs: 8

Baseline: DEE coefficient 0, seed 20260729.

Fair: DEE coefficient 100, training bandwidth 0.25, Huber delta 0.01,
preference threshold 3, seed 20261731. The apparently large coefficient is
paired with the paper's unscaled Huber term, whose magnitude is approximately
`delta` times an absolute gap outside its quadratic region.

Checkpoint paths and SHA-256:

- Baseline:
  `/NHNHOME/WORKSPACE/26msit001_A/optiml/aidep/artifacts/checkpoints/baseline_best.pt`
  (`e3d4e1436edf4db8da39b3b5e5fb74e452bdc4bda040cae5c7734459e569b2ec`)
- Fair:
  `/NHNHOME/WORKSPACE/26msit001_A/optiml/aidep/artifacts/checkpoints/fair_best.pt`
  (`6eb043832e2365ae0eb2e54b70b05c4c693b4b3a6fce858a836a731f4ddb847a`)

Selected YAML files are `configs/selected_baseline.yaml` and
`configs/selected_fair.yaml`. Every trial also has its exact config,
epoch-by-epoch JSONL log, summary, and best trial checkpoint under
`artifacts/tuning/`.

### Baseline reuse audit

No additional baseline training was necessary for the baseline website. The
existing selected checkpoint is a genuine pre-DEE model, not the fair model
under a different filename:

- embedded checkpoint config: `train.dee_coefficient = 0.0`;
- best epoch: 7 of 8, from random initialization;
- all eight `training.jsonl` epochs: `fairness_batches = 0`, `train_dee = 0`,
  and `train_dee_huber = 0`;
- training summary: `cuda:0`, exactly one visible GPU, 4,832 training users and
  604 validation users;
- selected checkpoint SHA-256 exactly matches
  `artifacts/tuning/baseline/base_64/best.pt`;
- evaluation exists for 604 disjoint cold-start test users at support sizes
  1, 3, 5, and 10 in `artifacts/metrics/baseline_test.json`.

The nonzero DEE numbers in baseline evaluation are post-training fairness
measurements; they do not indicate that DEE was included in its training loss.

## Website and inference

`src/coldstart_rec/inference.py` provides the reusable `MovieRecommender` API.
It computes a new user context in a forward pass and never changes checkpoint
weights. Rated items are excluded. Zero ratings use a learned null context.
Unknown/prefer-not-to-answer maps to the trained unknown gender token.

`app.py` provides the shared UI and defaults to the fair checkpoint.
`app_baseline.py` is a thin, nonduplicating entry point that selects
`baseline_best.pt`. Both provide:

- gender selection including unknown/prefer-not-to-answer;
- explicit confirmation on the results screen that the selected gender (or no
  gender information) and ratings were passed to inference;
- a guided two-step Korean interface: taste input, then recommendation results;
- popularity-ordered, searchable selection over the local 3,883-title catalog;
- no visible internal movie IDs;
- natural English article ordering in displayed titles and search (`Title, The`
  is shown as `The Title`) while canonical MovieLens titles remain unchanged;
- title, release year, and Korean genre labels during both selection and results;
- direct 1–5 star controls on a card for every selected watched movie;
- a visible progress indicator and an enforced minimum of three completed
  ratings before recommendation;
- 10 title/year/genre/predicted-score recommendation cards on a separate screen;
- a back action that preserves the entered ratings for editing;
- rated-item exclusion;
- a full session reset;
- an optional genre-distribution diagnostic;
- no external APIs or keys;
- no persistence of entered user data beyond Streamlit session state.

The underlying reusable API retains zero/few-rating handling. The web onboarding
requires three ratings by design so the user receives useful feedback before
moving to the results screen.

Launch the fair and baseline sites separately:

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/optiml/aidep
.venv/bin/streamlit run app.py --server.port 8501
.venv/bin/streamlit run app_baseline.py --server.port 8502
```

The verified example conditioned a female profile on ratings
`1:5, 260:5, 1196:5, 1721:2`. Its first five recommendations were:

1. *The Shawshank Redemption* (1994), 4.5756
2. *Sanjuro* (1962), 4.5562
3. *Seven Samurai* (1954), 4.5529
4. *Schindler's List* (1993), 4.5447
5. *The Usual Suspects* (1995), 4.5444

All ten are in `artifacts/example_recommendations.json`; none is a supplied
support movie.

The same profile was also run through the baseline checkpoint. Its first five
recommendations were:

1. *Seven Samurai* (1954), 4.5985
2. *The Shawshank Redemption* (1994), 4.5800
3. *Sunset Boulevard* (1950), 4.5725
4. *Schindler's List* (1993), 4.5623
5. *Paths of Glory* (1957), 4.5620

All ten baseline rows are in `artifacts/example_recommendations_baseline.json`;
none is a supplied support movie.

## Files created

Core modules:

- `src/coldstart_rec/data.py`: safe download parsing, serialization, split, episodes
- `src/coldstart_rec/model.py`: user-ID-free implicit and explicit
  gender-by-genre Wide-and-Deep conditional set encoders
- `src/coldstart_rec/fairness.py`: KDE DEE metric and Huber regularizer
- `src/coldstart_rec/groups.py`: paper genre-group mapping
- `src/coldstart_rec/episodes.py`: padded chronological task batches
- `src/coldstart_rec/training.py`: MSE, controlled stress, and DEE episodic optimization
- `src/coldstart_rec/evaluation.py`: quality, DEE, exposure, gender and
  counterfactual-interaction validation metrics
- `src/coldstart_rec/metrics.py`: regression/ranking/exposure metrics
- `src/coldstart_rec/checkpoint.py`: portable checkpoint format
- `src/coldstart_rec/inference.py`: stateless serving API
- `src/coldstart_rec/config.py`, `utils.py`: configuration and reproducibility

Entry points:

- `scripts/download_data.py`
- `scripts/train.py`
- `scripts/tune.py`
- `scripts/run_gender_stress_experiment.py`
- `scripts/evaluate.py`
- `scripts/audit_gender_effect.py`
- `scripts/recommend.py`
- `scripts/smoke_test_app.py`
- `app.py`
- `app_baseline.py`
- `app_stress_baseline.py`
- `app_stress_fair.py`

Tests:

- `tests/test_data.py`
- `tests/test_episodes.py`
- `tests/test_model.py`
- `tests/test_groups.py`
- `tests/test_fairness.py`
- `tests/test_inference.py`

Configuration/dependencies:

- `configs/explicit_*.yaml`, `configs/gender_stress_search.yaml`, and
  `configs/selected_{explicit_natural,stress_baseline,stress_control,stress_fair}.yaml`
- original `configs/*.yaml`
- `pyproject.toml`
- `requirements.txt`
- `README.md`

## Commands executed

The material reproducibility commands were:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install 'streamlit>=1.36,<2'
.venv/bin/python scripts/download_data.py --seed 20260729
.venv/bin/python -m pytest -q

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py \
  --config configs/smoke.yaml --output-dir artifacts/smoke/baseline
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py \
  --config configs/smoke_fair.yaml --output-dir artifacts/smoke/fair \
  --initial-checkpoint artifacts/smoke/baseline/best.pt

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/tune.py \
  --baseline-config configs/baseline.yaml --fair-config configs/fair.yaml \
  --search-config configs/search.yaml --output-root artifacts

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

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_gender_stress_experiment.py \
  --natural-config configs/explicit_natural.yaml \
  --stress-config configs/explicit_stress.yaml \
  --fair-config configs/explicit_fair.yaml \
  --search-config configs/gender_stress_search.yaml \
  --output-root artifacts/gender_stress

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/baseline_best.pt \
  --data data/processed --output artifacts/metrics/baseline_test.json \
  --support-sizes 1 3 5 10
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate.py \
  --checkpoint artifacts/checkpoints/fair_best.pt \
  --data data/processed --output artifacts/metrics/fair_test.json \
  --support-sizes 1 3 5 10

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

CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/audit_gender_effect.py \
  --checkpoint artifacts/checkpoints/stress_baseline_best.pt \
  --data data/processed \
  --output artifacts/metrics/stress_baseline_gender_audit.json \
  --split test --support-sizes 1 3 5 10 --top-k 10 --batch-size 64 \
  --thresholds 3.0 3.5 4.0 --bandwidth 0.5 --seed 20260729 --device cuda
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/audit_gender_effect.py \
  --checkpoint artifacts/checkpoints/stress_fair_best.pt \
  --data data/processed \
  --output artifacts/metrics/stress_fair_gender_audit.json \
  --split test --support-sizes 1 3 5 10 --top-k 10 --batch-size 64 \
  --thresholds 3.0 3.5 4.0 --bandwidth 0.5 --seed 20260729 --device cuda

.venv/bin/python scripts/recommend.py \
  --checkpoint artifacts/checkpoints/fair_best.pt --gender female \
  --rating 1:5 --rating 260:5 --rating 1196:5 --rating 1721:2 \
  --top-k 10 --output artifacts/example_recommendations.json

.venv/bin/python scripts/recommend.py \
  --checkpoint artifacts/checkpoints/baseline_best.pt --gender female \
  --rating 1:5 --rating 260:5 --rating 1196:5 --rating 1721:2 \
  --top-k 10 --output artifacts/example_recommendations_baseline.json

.venv/bin/python scripts/smoke_test_app.py --app app.py \
  --expected-checkpoint artifacts/checkpoints/fair_best.pt
.venv/bin/python scripts/smoke_test_app.py --app app_baseline.py \
  --expected-checkpoint artifacts/checkpoints/baseline_best.pt
.venv/bin/python scripts/smoke_test_app.py --app app_stress_baseline.py \
  --expected-checkpoint artifacts/checkpoints/stress_baseline_best.pt \
  --expected-model-label '통제 편향 Stress Baseline · DEE 미적용' \
  --expected-notice '실제 성별에 따른 영화 취향을 대표하지 않으며'
.venv/bin/python scripts/smoke_test_app.py --app app_stress_fair.py \
  --expected-checkpoint artifacts/checkpoints/stress_fair_best.pt \
  --expected-model-label '통제 편향 완화 모델 · DEE 적용' \
  --expected-notice '실제 성별에 따른 영화 취향을 대표하지 않으며'
.venv/bin/python -m compileall -q src scripts app.py app_baseline.py \
  app_stress_baseline.py app_stress_fair.py
.venv/bin/python -m pytest -q
```

The final verification produced `29 passed`. All four entry points passed the
full two-screen smoke flow with three ratings, ten recommendations, correct
checkpoint identity, preserved edit state, reset, genre rendering, rated-item
exclusion, and no visible movie IDs. The two stress sites additionally passed
the warning check on both screens.

## Failures, assumptions, and unresolved limitations

- Direct system/user pip installation was rejected by PEP 668; the successful
  solution is a project-local `--system-site-packages` virtual environment,
  preserving the host's B200-compatible PyTorch build.
- The OpenReview supplementary archive for the DEE paper returned HTTP 403.
  The implementation therefore follows the published equations exactly but
  cannot resolve the paper's undocumented multi-genre ambiguity from code.
- A first sampled-validation tuning audit did not generalize to dense test DEE.
  It was explicitly rejected; validation was changed to exact full-catalog
  DEE at a common bandwidth and the bounded search was rerun. This is why only
  the latest `tuning_summary.json` should be used.
- The selected model is deterministic TaNP-inspired CNP/DeepSets, not full
  TaNP. It omits latent uncertainty, task clustering, and FiLM in exchange for
  maintainability and reliable zero/few-shot serving.
- The new demonstrator is an explicit Wide-and-Deep variant of that robust
  cold-start tower, not MeLU or a claim of exact reproduction of an external
  architecture. The explicit head was selected because its gender effect is
  interpretable and counterfactually replaceable.
- The large stress-baseline separation is deliberately created by
  `L_stress`; it is not a naturally discovered MovieLens fact. MovieLens has
  only binary M/F labels, which do not represent the full range of gender.
- Removing the stress objective and continuing MSE alone reduces mean DEE by
  47.39%, so the complete stress-to-fair change cannot be assigned to DEE.
- In the controlled experiment, score-DEE improves at thresholds 3.0, 3.5,
  and 4.0, but Ranking-DEE does not improve. Fair scores also shift upward
  (five-shot mean 3.3069 to 3.6110), so calibration remains unresolved.
- The selected stress and fair trials reached their best value at the final
  bounded epoch. They are the best within the predeclared budget, not evidence
  of complete convergence. Results use one split and one seed per trial.
- MovieLens items are warm. A brand-new movie not in the checkpoint cannot be
  recommended without catalog retraining.
- Only gender and entered ratings are user inputs. MovieLens age, occupation,
  and ZIP were parsed for provenance but intentionally not used.
- DEE at a threshold controls a binary soft preference rate; it does not
  guarantee equality of full score distributions, relevance, calibration, or
  individual fairness. It can also suppress genuine group preference signals.
- Unknown gender is supported for serving but absent from MovieLens and
  excluded from binary M/F DEE auditing.
- The paper-group rule excludes 2,471 both/neither movies from DEE, although
  they remain recommendable. This is a documented response to an ambiguity in
  the source paper.
- Top-10 exposure remains concentrated: several genres have zero exposure in
  the five-shot diagnostic. DEE improves joint preference-rate independence
  in the stress experiment but not its Ranking-DEE, and it is not a
  catalog-coverage objective.
- Ranking metrics use held-out rated candidates, so HitRate@10 saturates and is
  less informative than RMSE, Recall, and NDCG here. Full-catalog implicit
  ranking would require a defensible negative-sampling or exposure model.
- Results use one deterministic user split and one seed per trial, not repeated
  confidence intervals. The retained logs allow a multi-seed extension.
