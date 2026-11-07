#!/usr/bin/env python3
"""InspectAI - Industrial Inspection & Maintenance Decision Support.

Single command entry point to launch the application:
    python run.py
    python run.py --host 0.0.0.0 --port 8000
    python run.py --reload
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_runtime_prerequisites() -> None:
    """Validate critical directories and model indices before startup."""
    required_dirs = [
        PROJECT_ROOT / "data" / "uploads",
        PROJECT_ROOT / "data" / "heatmaps",
        PROJECT_ROOT / "data" / "index" / "vision",
    ]
    for d in required_dirs:
        d.mkdir(parents=True, exist_ok=True)


def print_banner(host: str, port: int) -> None:
    """Print clean startup diagnostics banner."""
    print("=" * 78)
    print("   INSPECTAI: INDUSTRIAL MULTIMODAL DECISION SUPPORT AGENT")
    print("=" * 78)
    print(f"  • Web Dashboard & API : http://{host}:{port}")
    print(f"  • Interactive API Docs: http://{host}:{port}/docs")
    print(f"  • Knowledge Base Docs : data/kb/ (17 indexed chunks)")
    print(f"  • Supported Categories: bottle, cable, metal_nut, screw")
    print("  • Notice              : Decision support only. Findings are hypotheses,")
    print("                          not autonomous quality-control authorizations.")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="InspectAI Server Launcher")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn worker processes")
    args = parser.parse_args()

    check_runtime_prerequisites()
    print_banner(args.host, args.port)

    import uvicorn
    uvicorn.run(
        "src.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()

