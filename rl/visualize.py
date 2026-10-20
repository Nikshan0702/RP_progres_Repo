"""
rl/visualize.py
===============
Command-line entry point to launch the RL Audio Processing Visualization Studio.

Usage
-----
# Start interactive web dashboard (default: http://127.0.0.1:8080)
python3 rl/visualize.py

# Specify custom port
python3 rl/visualize.py --port 8085

# Headless / no browser
python3 rl/visualize.py --no-browser

# Run in simulated demo mode
python3 rl/visualize.py --mode demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.visualization.config import DEFAULT_HOST, DEFAULT_PORT
from rl.visualization.server import VisualizationRequestHandler, run_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch DAPN RL Audio Processing Visualization Studio"
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host address to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind (default: 8080)")
    parser.add_argument("--mode", type=str, choices=["real", "demo"], default=None, help="Execution mode override")
    parser.add_argument("--dry", type=str, default=None, help="Path to initial dry WAV file")
    parser.add_argument("--wet", type=str, default=None, help="Path to initial wet reference WAV file")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")
    args = parser.parse_args()

    # Pre-configure mode if specified
    if args.mode:
        VisualizationRequestHandler.engine.mode = args.mode

    if args.dry and args.wet:
        try:
            VisualizationRequestHandler.engine.initialize_session(
                dry_path=args.dry,
                wet_path=args.wet,
                mode=args.mode or "real",
            )
        except Exception as e:
            print(f"Warning: Failed to pre-load audio pair ({args.dry}, {args.wet}): {e}")

    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
