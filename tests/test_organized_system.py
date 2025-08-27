#!/usr/bin/env python3
"""
Test Organized System
====================

Test the reorganized Clipppy system to ensure all imports and components work correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all main components can be imported"""
    print("🧪 Testing Organized System Imports")
    print("=" * 50)
    
    # Test main CLI
    try:
        from twitch_clip_bot import TwitchClipBot
        print("✅ Main TwitchClipBot imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TwitchClipBot: {e}")
        return False
    
    # Test clip enhancer
    try:
        from src.clip_enhancer import ClipEnhancer
        print("✅ ClipEnhancer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ClipEnhancer: {e}")
        return False
    
    # Test TikTok uploader
    try:
        from services.tiktok_uploader import TikTokUploader
        print("✅ TikTokUploader imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TikTokUploader: {e}")
        return False
    
    # Test dashboard
    try:
        from services.dashboard import ClippyDashboard
        print("✅ ClippyDashboard imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ClippyDashboard: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading"""
    print("\n🔧 Testing Configuration System")
    print("-" * 30)
    
    try:
        from twitch_clip_bot import TwitchClipBot
        bot = TwitchClipBot()
        
        print(f"✅ Configuration loaded from: {bot.config_path}")
        print(f"✅ Enabled streamers: {len(bot.get_enabled_streamers())}")
        print(f"✅ TikTok uploader: {'Ready' if bot.tiktok_uploader else 'Not available'}")
        print(f"✅ Clip enhancer: {'Ready' if bot.clip_enhancer else 'Not available'}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_file_structure():
    """Test that all expected directories and files exist"""
    print("\n📁 Testing File Structure")
    print("-" * 25)
    
    expected_structure = {
        "src/": ["__init__.py", "clip_enhancer.py"],
        "services/": ["__init__.py", "tiktok_uploader.py", "dashboard.py"],
        "config/": ["config.yaml"],
        "docs/": ["README.md", "PROJECT_STRUCTURE.md"],
        "": ["clipppy.py", "twitch_clip_bot.py", "requirements.txt", "setup.py"]
    }
    
    all_good = True
    for directory, files in expected_structure.items():
        for file_name in files:
            file_path = Path(project_root / directory / file_name)
            if file_path.exists():
                print(f"✅ {directory}{file_name}")
            else:
                print(f"❌ Missing: {directory}{file_name}")
                all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("🎬 CLIPPPY ORGANIZED SYSTEM TEST")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test imports
    if test_imports():
        tests_passed += 1
    
    # Test configuration
    if test_configuration():
        tests_passed += 1
    
    # Test file structure
    if test_file_structure():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Organized system is working correctly")
        print("🚀 Ready for production use!")
        return True
    else:
        print("❌ Some tests failed")
        print("💡 Please check the error messages above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
