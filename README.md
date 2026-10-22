# DASE7506 MP2 - Selecting Reliable LLM Reasoning

Student ID: **3036837382**. Protocol: `dase7506-mp2-open-hard-verifier-v5`.

**Official full-test score: 79.65738758029978% (372/467).** Validation: 731/798,
91.604010%. The submitted model is unchanged from the audited checkpoint.

Method: one shared L2-regularised logistic-regression scorer over group-centred
normalised embeddings and five answer-consensus features; 1,029 dimensions,
`C=0.002`, seed 42. The two inference files total **26,244 B**. Original full-test
prediction plus independent scoring averaged 0.199 seconds on Apple M2; the
independent-directory audit measured 0.275 seconds. These are local measurements;
the course reference Xeon determines runtime eligibility.

## 1. Version and contents

Use the final source tag **`mp2-final-20261022`** with
`release/checkpoint/model_bundle/`, not a later development branch. Exact hashes
and the version mapping are in `release/version.json` and `release/asset-hashes.txt`.
The pinned upstream course release is `xudongwu-0/DASE7506-MP2` at
`bb801f4f2e7629d270df765ab12c333b2dc616bb`.

| Location | Purpose |
|---|---|
| `code/`, `guide/` | Original course interface, evaluator and guide; manifest files unchanged |
| `method/features.py`, `method/submission.py` | Input-only features and frozen NumPy predictor |
| `method/train.py` | Training, preprocessing and bundle export |
| `method/train_group.py` | Corrected optional group-loss comparison |
| `experiments.csv` | Main historical search, including reruns |
| `experiments_v1_unstandardized.csv` | Separate earlier preprocessing version |
| `runs/corrections/` | Diagnostic runs, gradient checks and reproduction evidence |
| `release/` | Submitted checkpoint, score, resource and experiment summaries |
| `report/report.pdf` | Final report, at most ten pages |

## 2. Installation

Run from this repository root with **Python 3.12**:

```bash
python3.12 -m venv code/.venv
source code/.venv/bin/activate
python -m pip install -r code/requirements.txt
python -m pip install -r requirements-method.txt
python tools/fetch_official_data.py
python code/check_data.py
```

The data helper downloads only official data from the pinned release and verifies
every supplied data hash before writing. Matching existing data is left untouched.
Alternatively copy that release's `code/data/` directory into this repository.
Data is not committed here. Its license/notice are retained by the download helper.
Prediction and scoring work offline after installation. Frozen inference needs
only NumPy; SciPy and scikit-learn are pinned for training.

Set thread limits for subsequent commands:

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4
```

Use one four-thread numerical process at a time. On Windows, set equivalent
environment variables before invoking the same Python commands.

## 3. Reproduce without retraining

From the repository root:

```bash
python code/predict.py --method method/submission.py \
  --checkpoint release/checkpoint/model_bundle --split test \
  --output runs/reproduced-test-predictions.json
python code/score_predictions.py --split test \
  --predictions runs/reproduced-test-predictions.json \
  --output runs/reproduced-test-score.json
```

Expected: `correct=372`, `n=467`, `accuracy=79.65738758029978`. These commands do
not overwrite the submitted score or checkpoint. Verify hashes, interfaces,
exact saved choices and complete fresh-process timing in an independent source copy:

```bash
python tools/verify_release.py --output runs/my-release-verification.json
```

This uses the installed Python environment, not a fresh dependency installation.
Its JSON separates prediction/scoring wall time and records thread caps.

## 4. Retrain from scratch

Fit only the official training split, writing a new bundle:

```bash
python method/train.py --config rel_only_vote --C 0.002 --seed 42 \
  --bundle runs/retrained-model --report runs/retrained-model.json
python code/predict.py --method method/submission.py \
  --checkpoint runs/retrained-model --split validation \
  --output runs/retrained-validation-predictions.json
python code/score_predictions.py --split validation \
  --predictions runs/retrained-validation-predictions.json \
  --output runs/retrained-validation-score.json
```

Expected validation: 731/798. The coefficients, intercept, means and standard
deviations were reproduced exactly in the audit environment. Another BLAS/platform
may introduce small numerical differences. Use the original assets, including
their original metadata, to reproduce their published hashes.

## 5. Experiments, corrections and analysis

```bash
python tools/mp2_stats.py
python tools/analyze.py --checkpoint release/checkpoint/model_bundle
python tools/test_contract.py
python tools/test_group_gradient.py
python tools/repair_controls.py --out runs/my-correction-reproduction
python tools/summarize_experiments.py
python tools/make_tables.py
```

The correction runner refuses to overwrite existing records and caps all its
numerical libraries at one thread. It repeats the corrected group comparison
(`lambda=0.001`, learning rate 0.05, 1,500 iterations, seed 42), direct rel-only and
vote-only removals at the frozen `C=0.002`, and a linear-centring equivalence check.
Saved results:

| Model | Validation correct / 798 | Accuracy % |
|---|---:|---:|
| Frozen rel+vote | 731 | 91.6040 |
| Rel-only, fixed C | 725 | 90.8521 |
| Vote-only, fixed C | 688 | 86.2155 |
| Corrected group loss, best checked step 50 | 725 | 90.8521 |

These controls were run after the test result was known, using train/validation
only. They did not reselect or replace the submitted method. The original group
gradient divided by 4N instead of N; its old result is retained as a historical
implementation record, not evidence against the stated loss. Seven mathematical
checks cover finite differences, L2, all-positive groups and extreme scores.

To repeat historical search families (appends to `experiments.csv`):

```bash
python tools/sweep.py --configs rel_only_vote \
  --C 0.0003,0.001,0.0015,0.002,0.003,0.005 --seeds 17,42,2026 --note rerun
python tools/sweep.py --configs m1_l2,m2_rel,m2_vote,m2_full,ablate_norel,ablate_novote \
  --C 0.003,0.01,0.1,1 --note rerun
python tools/sweep.py --configs rel_only_vote --C 0.002 --mlp 64,128 --note rerun
```

There are 125 main CSV rows (120 distinct IDs), 12 legacy rows, one historical
group run and the original final build: **270.549 seconds of logged fitting**.
Corrections add **24.748 seconds**. This is not development wall time; shared
feature times cannot be summed per CSV row. Rerunning the same ID appends CSV but
overwrites its old JSON/bundle. `release/experiment-summary.json` preserves these
cost categories and explains what cannot be reconstructed.

## 6. Interpretation, provenance and AI disclosure

With fixed linear weights, subtracting the same group mean adds a common score
offset and cancels in candidate ranking. Centred training changes fitted scales,
regularisation and weights; its better validation result does not establish
dynamic semantic comparison at inference. Vote features do compare answers.
Explicit surface features add no measured benefit, but embeddings may encode
format. MLP training 100% is selection accuracy, not label memorisation.

Current code fits weights and preprocessing statistics only on training data;
validation is used for selection. Prediction uses input and checkpoint only, not
reference answers or IDs. Available history does not independently prove the
original test-access/freeze chronology or pre-registration; no retrospective
guarantee is claimed. Corrections preserve final asset hashes and test results.

AI assistance implemented code, ran experiments, drafted the report and helped
audit and correct it. Saved logs and official scorers support numerical claims.
The student remains responsible for reviewing the implementation and claims.
GSM8K and Qwen3-0.6B attribution and the supplied license/notice are retained.
