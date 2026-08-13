# Artifact Map

The resume source of truth is under `research/`:

- `research/benchmark/` contains the 5,000-session four-strategy benchmark.
- `research/ppo/` contains the 200,000-step PPO model and 1,000-session holdout.
- `research/resume_metrics_verification.md` derives the approved metrics.
- `research/SHA256SUMS` records checksums for the final reports, row-level data,
  summaries, saved PPO model, and resume-verification document.

The CSV and JSON files directly under `artifacts/`, plus `verification/`, are
earlier two-strategy and PPO smoke checks retained for provenance. They should
not be used for current resume claims.

`research/smoke-benchmark/`, `research/calibration-400/`, and
`research/ppo-smoke/` are development checks. The final benchmark and holdout
directories are the only results referenced by the current documentation.
