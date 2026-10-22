# Corrections and release identity - 2026-10-22

The final predictor and both checkpoint files remain byte-for-byte identical to
the audited release. The official test result is still **372/467 =
79.65738758029978%**. No corrective result selected a different submission.

Changes:

- Corrected the optional group objective's data gradient from a 4N-row average to
  an N-question average. Stable log-sum-exp also handles extreme scores.
- Added seven synthetic mathematical checks and reran the original group
  configuration using train/validation only. Best checked step 50: 725/798.
- Added direct, fixed-C removals: rel-only 725/798 and vote-only 688/798, versus
  frozen rel+vote 731/798. These are post-test-result diagnostic controls.
- Corrected group-centring interpretation, explicit-format controls, MLP metric,
  vote-pattern labels and the 443-question single-valid-answer subset.
- Removed retrospective guarantees of historical pre-registration/test chronology
  that the original two-commit history cannot independently establish.
- Summarised separate historical CSV versions, reruns, group and final fits.
- Added a pinned data installer and an independent-directory release verifier.

Evidence: `runs/corrections/repair_controls_manifest.json`, the accompanying logs,
`release/experiment-summary.json`, and
`runs/corrections/release_verification.json`.

The final source/report tag is `mp2-final-20261022`; use it with the exact assets
listed in `release/version.json`. Local CPU timing is not a measurement on the
course reference Xeon. The independent-copy check uses the installed environment.
