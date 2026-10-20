"""Deprecated: replaced by ``rl.evaluate_protocol``.

The previous benchmark loaded legacy (pre-repair) checkpoints, compared methods on differing
baselines (dry vs predictor) and clamped negative improvements to zero. The legacy implementation is
kept for reference as ``rl/benchmark_comparison_legacy.py`` (not importable-safe, not maintained).
Use:
    python -m rl.evaluate_protocol --help
"""
import sys

if __name__ == "__main__":
    sys.exit("rl.benchmark_comparison is deprecated; use `python -m rl.evaluate_protocol`.")
