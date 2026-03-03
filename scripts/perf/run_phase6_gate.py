"""Run the phase6 benchmark with an optional automatic regression baseline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase6 perf gate using run_benchmark.py")
    parser.add_argument("--config", default="config.yaml", help="Config file for main.py")
    parser.add_argument("--seconds", type=int, default=60, help="Benchmark duration seconds")
    parser.add_argument("--output-dir", default=str(Path("artifacts") / "perf"), help="Perf artifacts directory")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="API base URL")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Telemetry poll interval seconds")
    parser.add_argument("--python-exe", default=sys.executable, help="Python interpreter")
    parser.add_argument("--compare-to", default="", help="Explicit baseline report.json path")
    parser.add_argument("--no-auto-baseline", action="store_true", help="Disable auto baseline discovery")
    parser.add_argument(
        "--max-metadata-lag-p95-ms",
        type=float,
        default=600.0,
        help="Maximum allowed p95 metadata lag in milliseconds",
    )
    parser.add_argument(
        "--max-janus-ttff-p95-ms",
        type=float,
        default=8000.0,
        help="Maximum allowed p95 Janus time-to-first-frame in milliseconds",
    )
    parser.add_argument(
        "--max-overlay-jank-ratio-p95",
        type=float,
        default=0.40,
        help="Maximum allowed p95 overlay jank ratio (0..1)",
    )
    parser.add_argument(
        "--min-overlay-fps-avg",
        type=float,
        default=8.0,
        help="Minimum allowed average overlay FPS",
    )
    parser.add_argument(
        "--p95-regression-tolerance-ms",
        type=float,
        default=5.0,
        help="Allowed p95 increase (ms) versus baseline",
    )
    parser.add_argument(
        "--fps-regression-tolerance",
        type=float,
        default=0.5,
        help="Allowed FPS drop versus baseline",
    )
    parser.add_argument("--skip-api", action="store_true", help="Do not start API during benchmark")
    parser.add_argument("--allow-existing-service", action="store_true", help="Allow existing service on base URL")
    return parser.parse_args()


def discover_latest_report(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    reports: List[Path] = sorted(
        output_dir.glob("*/report.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return reports[0] if reports else None


def main() -> int:
    args = parse_args()
    benchmark_script = Path(__file__).with_name("run_benchmark.py").resolve()
    output_dir = Path(args.output_dir).resolve()
    compare_to = str(args.compare_to).strip()
    baseline: Optional[Path] = None

    if compare_to:
        candidate = Path(compare_to).resolve()
        if candidate.exists() and candidate.is_file():
            baseline = candidate
    elif not args.no_auto_baseline:
        baseline = discover_latest_report(output_dir)

    cmd = [
        str(args.python_exe),
        str(benchmark_script),
        "--config",
        str(args.config),
        "--seconds",
        str(int(args.seconds)),
        "--output-dir",
        str(output_dir),
        "--base-url",
        str(args.base_url),
        "--poll-interval",
        str(float(args.poll_interval)),
        "--max-metadata-lag-p95-ms",
        str(float(args.max_metadata_lag_p95_ms)),
        "--max-janus-ttff-p95-ms",
        str(float(args.max_janus_ttff_p95_ms)),
        "--max-overlay-jank-ratio-p95",
        str(float(args.max_overlay_jank_ratio_p95)),
        "--min-overlay-fps-avg",
        str(float(args.min_overlay_fps_avg)),
        "--p95-regression-tolerance-ms",
        str(float(args.p95_regression_tolerance_ms)),
        "--fps-regression-tolerance",
        str(float(args.fps_regression_tolerance)),
    ]

    if baseline is not None:
        cmd.extend(["--compare-to", str(baseline)])
        print(f"Using regression baseline: {baseline}")
    else:
        print("No regression baseline found; running quality gates only.")

    if args.skip_api:
        cmd.append("--skip-api")
    if args.allow_existing_service:
        cmd.append("--allow-existing-service")

    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
