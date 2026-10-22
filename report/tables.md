# MP2 result tables (auto-generated from experiments.csv and official scorers)

## Table 1 — Baselines and frozen model (official scorer)

| method | validation acc % | test acc % | note |
|---|---:|---:|---|
| B0 uniform random (seed 42) | 76.817 | 49.679 | fixed rule |
| B1 always candidate 0 | 78.195 | 49.679 | fixed rule |
| B2 majority vote over parsed answers | 86.216 | 69.165 | fixed rule |
| B3 first parseable candidate | 81.704 | 52.677 | fixed rule |
| **Frozen model (relative+vote, C=0.002)** | **91.604** | **79.657** | linear, dim 1029 |

## Table 2 — Feature-group comparison (validation, best C per row)

| config | features | dim | best C | val acc % | train acc % |
|---|---|---:|---:|---:|---:|
| m0_voteonly | vote | 5 | 0.003 | 86.216 | 85.258 |
| m0_surfonly | surf | 10 | 0.1 | 81.203 | 78.318 |
| m1_raw | raw | 1024 | 0.003 | 89.599 | 90.389 |
| m1_l2 | l2 | 1024 | 0.1 | 89.724 | 91.125 |
| m2_cos | l2+cos | 1026 | 0.003 | 89.975 | 90.242 |
| m2_rel | l2+rel | 2048 | 0.1 | 90.727 | 93.817 |
| m2_vote | l2+vote | 1029 | 1.0 | 90.476 | 90.894 |
| ablate_norel | l2+cos+vote | 1031 | 1.0 | 90.602 | 91.104 |
| rel_only_vote | rel+vote | 1029 | 0.002 | 91.604 | 92.240 |
| rel_cos_vote | rel+cos+vote | 1031 | 0.0015 | 91.604 | 92.050 |
| m2_full | l2+rel+cos+vote | 2055 | 0.003 | 91.479 | 92.702 |
| rel_surf_vote | rel+vote+surf | 1039 | 0.0015 | 91.604 | 92.072 |
| m3_rel_surf | l2+rel+cos+vote+surf | 2065 | 0.003 | 91.353 | 92.723 |
| m4_full | l2+rel+cos+vote+surf | 2065 | 0.002 | 91.353 | 92.408 |

## Table 3 - Direct feature ablations (fixed C=0.002, seed=42; validation)

These post-audit controls were run after the test result was known. They explain the frozen model; they do not select a replacement.

| comparison | features | dim | val acc % | delta vs frozen |
|---|---|---:|---:|---:|
| frozen | rel+vote | 1029 | 91.604 | +0.000 |
| remove vote | rel | 1024 | 90.852 | -0.752 |
| remove relative vector | vote | 5 | 86.216 | -5.388 |

Table 2 is a feature-combination comparison with independently selected C, not a drop-one-group ablation.

## Table 4 — Regularisation sensitivity of the frozen feature set (rel+vote)

| C | val acc % (seed 17/42/2026) |
|---|---|
| 0.0003 | 90.602 |
| 0.001 | 91.353 / 91.353 / 91.353 |
| 0.0015 | 91.604 / 91.604 / 91.604 |
| 0.002 | 91.604 / 91.604 / 91.604 |
| 0.003 | 91.479 / 91.479 / 91.479 |
| 0.005 | 91.353 / 91.353 / 91.353 |
| 0.01 | 91.103 |

## Table 5 — Nonlinear and group-loss alternatives (validation)

| method | val acc % | note |
|---|---:|---|
| MLP hidden=64 | 91.604 | adam, 400 iters |
| MLP hidden=128 | 91.228 | adam, 400 iters |
| corrected group multi-positive loss | 90.852 | group-mean gradient; best validation iteration 50; post-audit control |
| linear logreg (frozen) | 91.604 | reference |

The historical group result in runs/exp_group_relvote.json is retained as a record of an incorrect gradient implementation, not evidence about the stated objective. MLP train accuracy is question-level selection accuracy.
