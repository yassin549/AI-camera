# Performance Workflow

This document tracks how to run Phase 1 and Phase 6 performance checks.

## Phase 1: Target Check

1. Start backend:

```powershell
.\run_prod.bat
```

2. Wait at least 10-15 seconds for rolling gauges to populate, then run:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B scripts\perf\check_targets.py --base-url http://127.0.0.1:8080 --wait-seconds 20 --poll-interval 1
```

Targets live in:

- `scripts/perf/phase1_targets.json`

## Phase 6: Repeatable Benchmark + Regression Gates

Run benchmark and create comparable artifacts:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B scripts\perf\run_benchmark.py --config config.yaml --seconds 60 --poll-interval 1 --output-dir artifacts\perf
```

Run the one-command phase gate (auto-baseline + quality + regression checks):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -B scripts\perf\run_phase6_gate.py --config config.yaml --seconds 60 --output-dir artifacts\perf
```

Artifacts are written to a timestamped folder:

- `benchmark_config.yaml` (exact config used)
- `benchmark.log` (runtime output)
- `timings.csv` (per-frame stage timings)
- `report.json` (machine-readable summary with p50/p95 + transport samples + metadata lag)

Report schema is stable under:

- `schema_version: "phase6.v1"`

Notes:
- The harness now performs quality checks and exits non-zero on invalid runs (for example, no detector/tracker frames or API/capture never ready).
- The harness computes metadata lag, Janus time-to-first-frame, and overlay FPS/jank summaries.
- It fails when p95 latency budgets are exceeded or when p95 values regress versus a baseline report (`--compare-to`).
- If an existing backend is already bound on `--base-url`, the harness exits early unless `--allow-existing-service` is provided.
- To keep artifacts for debugging while not failing CI/local script flow, pass `--allow-invalid`.
