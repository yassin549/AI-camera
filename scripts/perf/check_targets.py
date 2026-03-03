"""Phase 1 performance target checker against the live /metrics endpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check runtime metrics against performance targets")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="API base URL (default: %(default)s)")
    parser.add_argument(
        "--targets",
        default=str(Path("scripts") / "perf" / "phase1_targets.json"),
        help="Path to target definitions JSON",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=15.0,
        help="Maximum wait for required gauges to appear (default: %(default)s)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval while waiting for gauges (default: %(default)s)",
    )
    return parser.parse_args()


def load_targets(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def fetch_metrics(base_url: str) -> Dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/metrics"
    req = urllib.request.Request(endpoint, method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid /metrics payload: expected object")
    return payload


def compare(lhs: float, op: str, rhs: float) -> bool:
    if op == ">=":
        return lhs >= rhs
    if op == "<=":
        return lhs <= rhs
    if op == ">":
        return lhs > rhs
    if op == "<":
        return lhs < rhs
    if op == "==":
        return lhs == rhs
    raise ValueError(f"Unsupported operator: {op}")


def evaluate_metric_targets(
    gauges: Dict[str, Any], metric_targets: List[Dict[str, Any]]
) -> Tuple[List[str], List[str], List[str]]:
    passed: List[str] = []
    failed: List[str] = []
    missing: List[str] = []

    for target in metric_targets:
        metric = str(target.get("metric", ""))
        name = str(target.get("name", metric))
        op = str(target.get("op", ""))
        expected = float(target.get("value", 0.0))
        unit = str(target.get("unit", "")).strip()
        unit_suffix = f" {unit}" if unit else ""
        if metric not in gauges:
            missing.append(f"{name}: gauge '{metric}' not available")
            continue
        actual = float(gauges[metric])
        ok = compare(actual, op, expected)
        line = f"{name}: {actual:.2f}{unit_suffix} {op} {expected:.2f}{unit_suffix}"
        if ok:
            passed.append(line)
        else:
            failed.append(line)
    return passed, failed, missing


def main() -> int:
    args = parse_args()
    targets_doc = load_targets(args.targets)
    targets = list(targets_doc.get("targets", []))
    metric_targets = [t for t in targets if str(t.get("source", "metrics")) == "metrics"]
    manual_targets = [t for t in targets if str(t.get("source", "")) == "manual"]

    deadline = time.time() + max(0.0, float(args.wait_seconds))
    last_payload: Dict[str, Any] | None = None
    last_error: str | None = None

    required_metric_names = {str(t.get("metric", "")) for t in metric_targets}
    while time.time() <= deadline:
        try:
            payload = fetch_metrics(args.base_url)
            last_payload = payload
            gauges = payload.get("gauges", {}) if isinstance(payload.get("gauges"), dict) else {}
            available = set(map(str, gauges.keys()))
            if required_metric_names.issubset(available):
                break
            last_error = None
        except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(max(0.1, float(args.poll_interval)))

    if last_payload is None:
        print("FAIL: could not read /metrics")
        if last_error:
            print(f"Reason: {last_error}")
        return 2

    gauges = last_payload.get("gauges", {}) if isinstance(last_payload.get("gauges"), dict) else {}
    passed, failed, missing = evaluate_metric_targets(gauges, metric_targets)

    print(f"Phase: {targets_doc.get('phase', 'unknown')} | Target file: {args.targets}")
    print(f"Metrics endpoint: {args.base_url.rstrip('/')}/metrics")
    print("")

    for line in passed:
        print(f"PASS  {line}")
    for line in failed:
        print(f"FAIL  {line}")
    for line in missing:
        print(f"MISS  {line}")

    if manual_targets:
        print("")
        print("MANUAL TARGETS (not auto-evaluated):")
        for target in manual_targets:
            name = str(target.get("name", "unnamed"))
            metric = str(target.get("metric", "unknown"))
            op = str(target.get("op", ""))
            value = float(target.get("value", 0.0))
            unit = str(target.get("unit", "")).strip()
            unit_suffix = f" {unit}" if unit else ""
            print(f"TODO  {name}: {metric} {op} {value:.2f}{unit_suffix}")

    if failed or missing:
        print("")
        print("Result: FAIL")
        return 2

    print("")
    print("Result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

