"""Legacy compatibility wrapper for phase6 perf gate."""

from __future__ import annotations

from run_phase6_gate import main


if __name__ == "__main__":
    raise SystemExit(main())
