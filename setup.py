#!/usr/bin/env python3
"""
Clipppy Setup Script
===================

Simple setup script for development and deployment.
"""

import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "clips/temp",
        "clips/fonts", 
        "data",
        "logs",
        "uploads"
    ]
    
    print("📁 Creating necessary directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print("✅ Directories created successfully!")

def check_config():
    """Check if configuration exists"""
    config_file = Path("config/config.yaml")
    if config_file.exists():
        print("✅ Configuration file found")
        return True
    else:
        print("⚠️  Configuration file not found")
        print("   Please set up config/config.yaml with your Twitch credentials")
        return False

def main():
    """Main setup function"""
    print("🎬 Clipppy Setup")
    print("=" * 50)
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create directories
    create_directories()
    
    # Check configuration
    config_exists = check_config()
    
    print("\n" + "=" * 50)
    if config_exists:
        print("🎉 Setup complete! You can now run:")
        print("   python clipppy.py config     # Check configuration")
        print("   python clipppy.py start      # Start monitoring")
        print("   python clipppy.py dashboard  # Open web dashboard")
    else:
        print("⚠️  Setup incomplete!")
        print("   Please configure config/config.yaml before running Clipppy")
    
    return True

if __name__ == "__main__":
    main()
