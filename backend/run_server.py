#!/usr/bin/env python
"""
XR2Text API Server

Starts the FastAPI server for X-ray report generation.

Usage:
    # Activate virtual environment first
    # Windows: swin\\Scripts\\activate
    # Linux/Mac: source swin/bin/activate

    # Run server
    python run_server.py

    # Run with custom port
    python run_server.py --port 8080

    # Run with checkpoint
    python run_server.py --checkpoint checkpoints/best_model.pt
"""

import os
import sys
import argparse
from pathlib import Path

import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Run XR2Text API Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )
    args = parser.parse_args()

    # Set checkpoint path as environment variable
    if args.checkpoint:
        os.environ["MODEL_CHECKPOINT"] = args.checkpoint

    print("=" * 60)
    print("XR2Text API Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Checkpoint: {args.checkpoint or 'Default (untrained)'}")
    print(f"Reload: {args.reload}")
    print("=" * 60)
    print(f"\nAPI Documentation: http://localhost:{args.port}/docs")
    print(f"Health Check: http://localhost:{args.port}/health")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
