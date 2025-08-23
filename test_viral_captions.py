#!/usr/bin/env python3

from clip_enhancer import ClipEnhancer
from pathlib import Path
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_viral_captions():
    print("🔥 TESTING VIRAL CAPTIONS - SIMPLIFIED VERSION")
    
    enhancer = ClipEnhancer()
    video_path = Path(r'C:\Users\samfi\AppData\Local\Temp\twitch_clips\enhanced_AgitatedDeterminedWasabiRuleFive-puM67stfNVwQflyr.mp4')
    
    if not video_path.exists():
        print(f"❌ File not found: {video_path}")
        return
    
    print(f"✅ Processing: {video_path.name}")
    
    # Test with a shorter clip to see captions faster
    try:
        result = enhancer.enhance_clip(
            video_path, 
            use_captions=True, 
            vertical_format=False,
            text_overlay="FALLBACK TEXT"  # In case captions fail
        )
        
        if result:
            print(f"🎬 VIRAL CLIP CREATED: {result}")
            print("📂 Check this file to see if captions are visible!")
        else:
            print("❌ Enhancement failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_viral_captions()
