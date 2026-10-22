# MP2 completion checklist

Status: 2026-10-22, UTC+8. Deadline: 2026-10-31 23:59.

- [x] Official manifest's 20 file hashes match; evaluator and data unchanged.
- [x] Multi-positive labels and training-only preprocessing verified.
- [x] Frozen checkpoint files unchanged; test score 372/467 = 79.65738758029978%.
- [x] Existing 15 interface checks pass; independent ID-invariance audit passes.
- [x] Corrected group gradient passes seven synthetic mathematical checks.
- [x] Original group configuration and fixed-setting direct feature removals rerun on train/validation only.
- [x] Feature-combination comparisons distinguished from direct ablations.
- [x] Mechanism, surface-feature, MLP metric and vote-pattern report claims corrected.
- [x] Search costs, legacy preprocessing and reruns separately summarised.
- [x] Checkpoint hashes, source tag and original audited commit recorded in release/version.json.
- [x] Independent-directory fresh-process verification exactly reproduces test predictions and score.
- [x] Assets total 26,244 B, below 16 MiB; local complete CPU timing is below 30 s.
- [x] Report rebuilt and visually checked; seven pages, including AI/reuse disclosure.
- [x] Tagged source and matching checkpoint published and independently accessible.
- [x] Encrypted artifact links submitted through the MP2 website; issue #243 and leaderboard listing confirmed.

Evidence is in runs/corrections/ and release/experiment-summary.json. The independent-copy
verification uses the installed Python environment, not a fresh package installation.
Runtime eligibility is decided on the course reference Xeon. Original pre-registration
and test-access chronology cannot be independently established by the saved history;
this revision makes no retrospective guarantee. All new controls are diagnostic and
preserve the original final model and test result.

Submission receipt: release/submission.json. Issue: https://github.com/xudongwu-0/xudongwu-0.github.io/issues/243
Leaderboard status is Self-reported, not instructor verification. The frozen source
tag is unchanged; this checklist and receipt are post-publication records on main.
