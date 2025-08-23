#!/usr/bin/env python3

import moviepy
from pathlib import Path

def create_viral_clip():
    print("🔥 MINIMAL VIRAL CLIP TEST")
    
    video_path = Path(r'C:\Users\samfi\AppData\Local\Temp\twitch_clips\enhanced_AgitatedDeterminedWasabiRuleFive-puM67stfNVwQflyr.mp4')
    
    # Load video
    video = moviepy.VideoFileClip(str(video_path))
    print(f"📐 Video: {video.size}, {video.duration}s")
    
    # Create viral captions exactly like working test
    captions = []
    
    # Caption 1: "FISHERMAN? NOT BAD, NOT GREAT!" 
    cap1 = moviepy.TextClip(
        text="🔥 FISHERMAN? NOT BAD, NOT GREAT! 🔥",
        font_size=80,
        color='yellow',
        stroke_color='black',
        stroke_width=4,
        duration=3.0
    ).with_position(('center', video.size[1] - 120)).with_start(0.9)
    
    captions.append(cap1)
    
    # Caption 2: "EXECUTIONER!"
    cap2 = moviepy.TextClip(
        text="EXECUTIONER! ⚡",
        font_size=75,
        color='red',
        stroke_color='black',
        stroke_width=4,
        duration=1.5
    ).with_position(('center', video.size[1] - 120)).with_start(6.1)
    
    captions.append(cap2)
    
    # Caption 3: "WIND CONDITION!"
    cap3 = moviepy.TextClip(
        text="🌪️ WIND CONDITION! 🌪️",
        font_size=85,
        color='green',
        stroke_color='black',
        stroke_width=4,
        duration=2.0
    ).with_position(('center', video.size[1] - 120)).with_start(6.9)
    
    captions.append(cap3)
    
    print(f"✅ Created {len(captions)} viral captions")
    
    # Composite everything
    all_clips = [video] + captions
    final_video = moviepy.CompositeVideoClip(all_clips)
    
    # Export
    output_path = Path(r'C:\Users\samfi\AppData\Local\Temp\twitch_clips\MINIMAL_VIRAL_TEST.mp4')
    final_video.write_videofile(str(output_path), codec='libx264', audio_codec='aac')
    
    print(f"🎬 VIRAL CLIP SAVED: {output_path}")
    
    # Cleanup
    video.close()
    final_video.close()
    for cap in captions:
        cap.close()

if __name__ == "__main__":
    create_viral_clip()
