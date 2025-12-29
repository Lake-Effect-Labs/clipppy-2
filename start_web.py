#!/usr/bin/env python3
"""
Start the Clipppy Web Application

Usage:
    python start_web.py              # Start web server only
    python start_web.py --all        # Start web server + orchestrator
    python start_web.py --init-db    # Initialize database only
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def init_database():
    """Initialize the database"""
    print("Initializing database...")
    from web.database import init_db
    init_db()
    print("Database initialized!")


def check_env():
    """Check required environment variables"""
    from dotenv import load_dotenv
    load_dotenv()

    required = ["TWITCH_CLIENT_ID", "TWITCH_CLIENT_SECRET"]
    missing = [var for var in required if not os.getenv(var)]

    if missing:
        print("WARNING: Missing required environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nTwitch OAuth won't work without these.")
        print("Set them in your .env file.\n")

    # Generate SECRET_KEY if not set
    if not os.getenv("SECRET_KEY"):
        import secrets
        key = secrets.token_hex(32)
        print(f"Generated SECRET_KEY (add to .env): SECRET_KEY={key}\n")


def start_web_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """Start the FastAPI web server"""
    print(f"Starting Clipppy web server at http://{host}:{port}")
    print("Press Ctrl+C to stop\n")

    import uvicorn
    uvicorn.run(
        "web.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def start_orchestrator():
    """Start the multi-tenant orchestrator"""
    import asyncio
    from services.orchestrator import MultiTenantOrchestrator

    print("Starting Multi-Tenant Orchestrator...")
    orchestrator = MultiTenantOrchestrator()
    asyncio.run(orchestrator.run())


def main():
    parser = argparse.ArgumentParser(description="Start Clipppy Web Application")
    parser.add_argument("--init-db", action="store_true", help="Initialize database only")
    parser.add_argument("--all", action="store_true", help="Start web server + orchestrator")
    parser.add_argument("--orchestrator", action="store_true", help="Start orchestrator only")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")

    args = parser.parse_args()

    # Always check env
    check_env()

    if args.init_db:
        init_database()
        return

    # Always init database
    init_database()

    if args.orchestrator:
        start_orchestrator()
    elif args.all:
        # Start both in separate processes
        import multiprocessing

        web_process = multiprocessing.Process(
            target=start_web_server,
            args=(args.host, args.port, not args.no_reload)
        )
        orchestrator_process = multiprocessing.Process(target=start_orchestrator)

        web_process.start()
        orchestrator_process.start()

        try:
            web_process.join()
            orchestrator_process.join()
        except KeyboardInterrupt:
            print("\nShutting down...")
            web_process.terminate()
            orchestrator_process.terminate()
    else:
        start_web_server(args.host, args.port, not args.no_reload)


if __name__ == "__main__":
    main()
