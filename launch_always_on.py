#!/usr/bin/env python3
"""
🚀 Always-On Clipppy Launcher
Starts the multi-streamer monitoring system that automatically detects when
configured streamers go live and spawns individual listeners for each.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Loaded .env file")
    else:
        print("⚠️ No .env file found")

# Load environment variables at startup
load_env_file()

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if config exists
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        print("❌ config/config.yaml not found")
        return False
    
    # Check environment variables
    required_vars = ['TWITCH_CLIENT_ID', 'TWITCH_CLIENT_SECRET']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("💡 Make sure to set these in your .env file or environment")
        return False
    
    # Check log directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    print("✅ All requirements met")
    return True

def show_status():
    """Show current system status"""
    print("\n" + "="*60)
    print("🚀 CLIPPPY ALWAYS-ON SYSTEM")
    print("="*60)
    print("📋 Features:")
    print("   • Monitors ALL configured streamers automatically")
    print("   • Spawns listeners when streamers go live")
    print("   • Concurrent clip enhancement")
    print("   • Centralized TikTok posting")
    print("   • 30-minute status check intervals")
    print("\n📱 TikTok Strategy:")
    print("   • Single account for all streamers")
    print("   • Auto-generated captions with credit")
    print("   • Game-specific hashtags")
    print("   • Optimal posting schedule")
    print("\n⚡ Performance:")
    print("   • Up to 3 concurrent enhancements")
    print("   • Automatic retry logic for downloads")
    print("   • Adaptive viral thresholds")
    print("="*60)

def main():
    """Main launcher"""
    show_status()
    
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues above.")
        return 1
    
    print("\n🚀 Starting Always-On Controller...")
    print("💡 Press Ctrl+C to stop\n")
    
    try:
        # Start the always-on controller
        process = subprocess.run([
            sys.executable, 'always_on_controller.py'
        ], cwd=os.getcwd())
        
        return process.returncode
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error starting controller: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
