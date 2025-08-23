#!/usr/bin/env python3

import moviepy
from pathlib import Path

def test_simple_caption():
    print("🧪 SIMPLE CAPTION TEST")
    
    video_path = Path(r'C:\Users\samfi\AppData\Local\Temp\twitch_clips\enhanced_AgitatedDeterminedWasabiRuleFive-puM67stfNVwQflyr.mp4')
    
    if not video_path.exists():
        print(f"❌ File not found: {video_path}")
        return
    
    print(f"✅ Loading video: {video_path}")
    
    # Load video
    video = moviepy.VideoFileClip(str(video_path))
    print(f"📐 Video size: {video.size}, duration: {video.duration}s")
    
    # Create a SUPER visible caption
    caption = moviepy.TextClip(
        text="🔥🔥🔥 CAPTION TEST 🔥🔥🔥",
        font_size=120,
        color='red',
        stroke_color='white',
        stroke_width=8,
        duration=5.0  # Show for 5 seconds
    ).with_position(('center', 'center'))  # Dead center of screen
    
    print("✅ Caption created")
    
    # Composite
    final_video = moviepy.CompositeVideoClip([video, caption])
    print("✅ Video composited")
    
    # Export
    output_path = Path(r'C:\Users\samfi\AppData\Local\Temp\twitch_clips\SIMPLE_CAPTION_TEST.mp4')
    final_video.write_videofile(str(output_path), codec='libx264', audio_codec='aac')
    
    print(f"🎬 SIMPLE TEST SAVED: {output_path}")
    
    # Cleanup
    video.close()
    final_video.close()
    caption.close()

if __name__ == "__main__":
    test_simple_caption()
