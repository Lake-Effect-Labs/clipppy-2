#!/usr/bin/env python3

from clip_enhancer import ClipEnhancer, CaptionWord, CaptionPhrase
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_captions():
    print("🔍 DEBUGGING CAPTION SYSTEM...")
    
    enhancer = ClipEnhancer()
    clip_path = Path(r'C:\Users\samfi\AppData\Local\Temp\twitch_clips\enhanced_AgitatedDeterminedWasabiRuleFive-puM67stfNVwQflyr.mp4')
    
    if not clip_path.exists():
        print(f"❌ File not found: {clip_path}")
        return
    
    print(f"✅ File exists: {clip_path}")
    
    # Test 1: Transcription
    print("\n🎤 Testing transcription...")
    words = enhancer.transcribe_audio(clip_path)
    
    if not words:
        print("❌ No words from transcription - creating fake data for testing")
        # Create fake transcription data for testing
        words = [
            CaptionWord("This", 0.0, 0.5, 1.0),
            CaptionWord("is", 0.5, 0.8, 1.0),
            CaptionWord("absolutely", 0.8, 1.5, 1.0),
            CaptionWord("insane", 1.5, 2.0, 1.0),
            CaptionWord("gameplay", 2.0, 2.8, 1.0),
            CaptionWord("right", 2.8, 3.1, 1.0),
            CaptionWord("here", 3.1, 3.5, 1.0),
        ]
        print(f"🎭 Using {len(words)} fake words for testing")
    else:
        print(f"✅ Got {len(words)} words from transcription")
        for i, word in enumerate(words[:5]):
            print(f"  Word {i+1}: '{word.text}' ({word.start_time:.1f}s - {word.end_time:.1f}s)")
    
    # Test 2: Enhancement
    print("\n📈 Testing word enhancement...")
    enhanced_words = enhancer.enhance_words_for_virality(words)
    emphasized_count = sum(1 for w in enhanced_words if w.is_emphasized)
    print(f"Enhanced: {emphasized_count} emphasized out of {len(enhanced_words)}")
    
    for word in enhanced_words:
        if word.is_emphasized:
            print(f"  🔥 EMPHASIZED: '{word.text}' -> {word.color} {word.emoji}")
    
    # Test 3: Phrase grouping
    print("\n📝 Testing phrase grouping...")
    phrases = enhancer.group_words_into_phrases(enhanced_words)
    print(f"Created {len(phrases)} phrases:")
    
    for i, phrase in enumerate(phrases):
        has_emphasis = any(w.is_emphasized for w in phrase.words)
        print(f"  Phrase {i+1}: '{phrase.text}' ({phrase.start_time:.1f}s-{phrase.end_time:.1f}s) {'🔥 VIRAL' if has_emphasis else ''}")
    
    # Test 4: Caption creation
    print("\n🎬 Testing caption creation...")
    if phrases:
        test_phrase = phrases[0]
        video_size = (1920, 1080)  # Assume HD video
        
        try:
            caption_clip = enhancer.create_animated_caption(test_phrase, video_size)
            if caption_clip:
                print(f"✅ Caption clip created successfully!")
                print(f"   Duration: {caption_clip.duration if hasattr(caption_clip, 'duration') else 'Unknown'}")
            else:
                print("❌ Caption clip creation returned None")
        except Exception as e:
            print(f"❌ Caption creation failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n🏁 Debug complete!")

if __name__ == "__main__":
    test_captions()
