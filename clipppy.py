#!/usr/bin/env python3
"""
Clipppy - TikTok Automation Pipeline
===================================

Main entry point for the Clipppy application.
Provides a clean interface to the CLI with proper import handling.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path for clean imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # Import the main CLI application
    from twitch_clip_bot import cli
    
    if __name__ == "__main__":
        cli()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from the project root directory")
    print("💡 Install dependencies with: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)
