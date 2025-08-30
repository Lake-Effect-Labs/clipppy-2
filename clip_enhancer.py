#!/usr/bin/env python3
"""
Clip Enhancer - AI-powered video editing for viral Twitch clips
"""

import os
import re
import requests
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
import tempfile
import logging

try:
    # Try direct import from moviepy (newer versions)
    import moviepy
    import moviepy.config as mpconf
    # Configure ImageMagick path for Windows
    mpconf.IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
    VideoFileClip = moviepy.VideoFileClip
    TextClip = moviepy.TextClip
    CompositeVideoClip = moviepy.CompositeVideoClip
    AudioFileClip = moviepy.AudioFileClip
    CompositeAudioClip = moviepy.CompositeAudioClip
    MOVIEPY_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        # Fallback to editor import for older versions
        from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, CompositeAudioClip
        import moviepy.config as mpconf
        # Configure ImageMagick path for Windows
        mpconf.IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
        MOVIEPY_AVAILABLE = True
    except ImportError:
        print("⚠️  MoviePy not installed. Run: pip install moviepy")
        VideoFileClip = TextClip = CompositeVideoClip = AudioFileClip = CompositeAudioClip = None
        MOVIEPY_AVAILABLE = False

try:
    import openai
except ImportError:
    print("⚠️  OpenAI not installed. Run: pip install openai")
    openai = None

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("⚠️  Whisper not installed. Run: pip install openai-whisper")
    whisper = None
    WHISPER_AVAILABLE = False

import math
import json
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class CaptionWord:
    """Represents a single word with timing and styling"""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    is_emphasized: bool = False
    color: str = 'white'
    emoji: str = ''


@dataclass
class CaptionPhrase:
    """Represents a phrase/sentence with multiple words"""
    words: List[CaptionWord]
    start_time: float
    end_time: float
    text: str


class ClipEnhancer:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        # Use local clips folder instead of system temp
        self.clips_dir = Path(__file__).parent / "clips"
        self.temp_dir = self.clips_dir / "temp"
        self.clips_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Sound effects library (you can add more)
        self.sound_effects = {
            'airhorn': 'https://www.soundjay.com/misc/sounds/bell-ringing-05.wav',
            'wow': 'https://www.soundjay.com/human/sounds/man-say-wow.wav',
            'epic': 'https://www.soundjay.com/misc/sounds/success-fanfare-trumpets.wav',
            'sus': 'https://www.soundjay.com/misc/sounds/fail-buzzer-02.wav'
        }
        
    def download_clip(self, clip_url: str) -> Optional[Path]:
        """Download a Twitch clip video file using direct API approach"""
        try:
            logger.info(f"📥 Attempting to download: {clip_url}")
            
            # Convert clip URL to download URL
            if '/edit' in clip_url:
                clip_url = clip_url.replace('/edit', '')
                logger.info(f"📝 Cleaned URL: {clip_url}")
            
            # Extract clip ID from URL
            url_parts = clip_url.split('/')
            if len(url_parts) < 2:
                logger.error(f"❌ Invalid URL format: {clip_url}")
                return None
            
            clip_id = url_parts[-1]
            if not clip_id or len(clip_id) < 10:  # Twitch clip IDs are much longer
                logger.error(f"❌ Could not extract clip ID from: {clip_url}")
                return None
                
            logger.info(f"🆔 Extracted clip ID: {clip_id}")
            
            # Try direct Twitch API approach first
            video_url = self._get_clip_video_url_from_api(clip_id)
            if video_url:
                logger.info(f"🎯 Got video URL from Twitch API: {video_url[:50]}...")
                return self._download_from_direct_url(video_url, clip_id)
            
            # Fallback to yt-dlp if API approach fails
            logger.warning("🔄 API approach failed, trying yt-dlp fallback...")
            return self._download_with_ytdlp(clip_url, clip_id)
            
        except Exception as e:
            logger.error(f"❌ Error downloading clip: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _get_clip_video_url_from_api(self, clip_id: str) -> Optional[str]:
        """Get direct video URL from Twitch API"""
        try:
            # Get Twitch credentials
            client_id = os.getenv('TWITCH_CLIENT_ID')
            client_secret = os.getenv('TWITCH_CLIENT_SECRET')
            oauth_token = os.getenv('TWITCH_OAUTH_TOKEN')
            
            if not client_id or not client_secret:
                logger.warning("⚠️ Missing Twitch credentials, skipping API approach")
                return None
            
            # Get access token
            access_token = oauth_token
            if not access_token:
                # Use client credentials
                auth_url = "https://id.twitch.tv/oauth2/token"
                auth_data = {
                    'client_id': client_id,
                    'client_secret': client_secret,
                    'grant_type': 'client_credentials'
                }
                auth_response = requests.post(auth_url, data=auth_data, timeout=10)
                auth_response.raise_for_status()
                access_token = auth_response.json()['access_token']
            
            # Get clip details from Twitch API
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Client-Id': client_id
            }
            
            url = f"https://api.twitch.tv/helix/clips?id={clip_id}"
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if not data.get('data'):
                logger.warning(f"⚠️ No clip data found for ID: {clip_id}")
                return None
            
            clip_data = data['data'][0]
            video_url = clip_data.get('video_url') or clip_data.get('thumbnail_url', '').replace('-preview-480x272.jpg', '.mp4')
            
            if video_url and video_url.endswith('.mp4'):
                logger.info(f"✅ Found direct video URL from API")
                return video_url
            
            logger.warning(f"⚠️ No direct video URL found in clip data")
            return None
            
        except requests.RequestException as e:
            logger.warning(f"⚠️ API request failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ API approach error: {e}")
            return None
    
    def _download_from_direct_url(self, video_url: str, clip_id: str) -> Optional[Path]:
        """Download video from direct URL"""
        try:
            output_path = self.temp_dir / f"{clip_id}.mp4"
            
            logger.info(f"🌐 Downloading from direct URL...")
            response = requests.get(video_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"✅ Downloaded clip via API: {output_path}")
                return output_path
            else:
                logger.error(f"❌ Downloaded file is empty or missing")
                return None
                
        except Exception as e:
            logger.error(f"❌ Direct download failed: {e}")
            return None
    
    def _download_with_ytdlp(self, clip_url: str, clip_id: str) -> Optional[Path]:
        """Fallback yt-dlp download method"""
        try:
            output_path = self.temp_dir / f"{clip_id}.mp4"
            cmd = ['yt-dlp', '--no-check-formats', '--output', str(output_path), clip_url]
            logger.info(f"🔧 Fallback yt-dlp: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"✅ Downloaded clip via yt-dlp fallback: {output_path}")
                return output_path
            else:
                logger.error(f"❌ yt-dlp fallback also failed")
                logger.error(f"❌ Error: {result.stderr[:300]}")
                return None
                
        except Exception as e:
            logger.error(f"❌ yt-dlp fallback error: {e}")
            return None
    
    def generate_viral_text(self, context: str = "") -> str:
        """Generate viral text overlay using AI"""
        if not self.openai_api_key or not openai:
            # Fallback viral texts
            fallback_texts = [
                "THAT'S INSANE! 🔥",
                "NO WAY! 😱",
                "CLIP IT! 🎬",
                "ABSOLUTELY CRACKED! ⚡",
                "HOLY! 🤯",
                "POGGERS! 🚀",
                "UNREAL! ✨",
                "GOATED! 👑"
            ]
            import random
            return random.choice(fallback_texts)
        
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            prompt = f"""
            Generate a short, viral, energetic text overlay for a Twitch gaming clip.
            Context: {context}
            
            Requirements:
            - Maximum 3 words
            - High energy and excitement
            - Gaming/streaming culture appropriate
            - Include 1-2 relevant emojis
            - Make it VIRAL and shareable
            
            Examples: "ABSOLUTELY CRACKED! ⚡", "NO SHOT! 🔥", "THAT'S INSANE! 😱"
            
            Generate ONE viral text:
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.9
            )
            
            text = response.choices[0].message.content.strip()
            return text
            
        except Exception as e:
            logger.error(f"AI text generation failed: {e}")
            return "EPIC CLIP! 🔥"
    
    def transcribe_audio(self, video_path: Path) -> Optional[List[CaptionWord]]:
        """Transcribe audio from video using Whisper"""
        try:
            logger.info("🎤 Transcribing audio...")
            
            # Try OpenAI Whisper API first
            if self.openai_api_key and openai:
                try:
                    client = openai.OpenAI(api_key=self.openai_api_key)
                    
                    with open(video_path, 'rb') as audio_file:
                        response = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="verbose_json",
                            timestamp_granularities=["word"]
                        )
                    
                    words = []
                    if hasattr(response, 'words') and response.words:
                        for word_data in response.words:
                            word = CaptionWord(
                                text=word_data.word.strip(),
                                start_time=word_data.start,
                                end_time=word_data.end,
                                confidence=1.0  # OpenAI doesn't provide confidence
                            )
                            words.append(word)
                    
                    logger.info(f"✅ OpenAI Whisper transcribed {len(words)} words")
                    return words
                    
                except Exception as e:
                    logger.warning(f"OpenAI Whisper failed: {e}, trying local Whisper...")
            
            # Fallback to local Whisper
            if WHISPER_AVAILABLE and whisper:
                try:
                    model = whisper.load_model("base")
                    result = model.transcribe(
                        str(video_path),
                        word_timestamps=True,
                        verbose=False
                    )
                    
                    words = []
                    for segment in result.get('segments', []):
                        for word_data in segment.get('words', []):
                            word = CaptionWord(
                                text=word_data['word'].strip(),
                                start_time=word_data['start'],
                                end_time=word_data['end'],
                                confidence=word_data.get('probability', 1.0)
                            )
                            words.append(word)
                    
                    logger.info(f"✅ Local Whisper transcribed {len(words)} words")
                    return words
                    
                except Exception as e:
                    logger.error(f"Local Whisper failed: {e}")
            
            logger.warning("No Whisper transcription available")
            return None
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None
    
    def enhance_words_for_virality(self, words: List[CaptionWord]) -> List[CaptionWord]:
        """Add emphasis, colors, and emojis to key words"""
        emphasis_words = {
            'insane', 'crazy', 'amazing', 'incredible', 'unbelievable', 'wow', 'omg',
            'sick', 'fire', 'cracked', 'goated', 'poggers', 'lets', 'go', 'holy',
            'absolutely', 'literally', 'actually', 'definitely', 'honestly', 'really'
        }
        
        positive_words = {
            'good', 'great', 'awesome', 'perfect', 'nice', 'clean', 'smooth', 'clutch'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'fail', 'miss', 'wrong', 'oof', 'rip'
        }
        
        emoji_map = {
            'fire': '🔥', 'sick': '🔥', 'insane': '🤯', 'crazy': '🤯', 'amazing': '✨',
            'incredible': '✨', 'wow': '😱', 'omg': '😱', 'good': '👍', 'great': '👍',
            'awesome': '🚀', 'perfect': '💯', 'nice': '👌', 'clean': '✨', 'clutch': '💪',
            'fail': '💀', 'rip': '💀', 'oof': '😬', 'lets': '🚀', 'go': '🚀'
        }
        
        enhanced_words = []
        for word in words:
            word_lower = word.text.lower().strip('.,!?')
            
            # Add emphasis and color
            if word_lower in emphasis_words:
                word.is_emphasized = True
                word.color = 'yellow'
            elif word_lower in positive_words:
                word.is_emphasized = True
                word.color = 'green'
            elif word_lower in negative_words:
                word.is_emphasized = True
                word.color = 'red'
            
            # Add emoji
            if word_lower in emoji_map:
                word.emoji = emoji_map[word_lower]
            
            enhanced_words.append(word)
        
        return enhanced_words
    
    def group_words_into_phrases(self, words: List[CaptionWord], max_phrase_duration: float = 3.0) -> List[CaptionPhrase]:
        """Group words into phrases for better readability"""
        if not words:
            return []
        
        phrases = []
        current_phrase_words = []
        phrase_start_time = words[0].start_time
        
        for word in words:
            # Start new phrase if duration exceeds limit or natural break
            if (current_phrase_words and 
                (word.start_time - phrase_start_time > max_phrase_duration or
                 word.text.endswith('.') or word.text.endswith('!') or word.text.endswith('?'))):
                
                if current_phrase_words:
                    phrase_text = ' '.join([w.text + w.emoji for w in current_phrase_words])
                    phrase = CaptionPhrase(
                        words=current_phrase_words,
                        start_time=phrase_start_time,
                        end_time=current_phrase_words[-1].end_time,
                        text=phrase_text
                    )
                    phrases.append(phrase)
                
                current_phrase_words = []
                phrase_start_time = word.start_time
            
            current_phrase_words.append(word)
        
        # Add final phrase
        if current_phrase_words:
            phrase_text = ' '.join([w.text + w.emoji for w in current_phrase_words])
            phrase = CaptionPhrase(
                words=current_phrase_words,
                start_time=phrase_start_time,
                end_time=current_phrase_words[-1].end_time,
                text=phrase_text
            )
            phrases.append(phrase)
        
        return phrases
    
    def create_animated_caption(self, phrase: CaptionPhrase, video_size: tuple) -> object:
        """Create an animated caption for a phrase - simplified for compatibility"""
        if not MOVIEPY_AVAILABLE:
            return None
        
        try:
            # Determine if any words are emphasized
            has_emphasis = any(word.is_emphasized for word in phrase.words)
            primary_color = 'yellow' if has_emphasis else 'white'
            
            # Use the color of the first emphasized word, or white
            for word in phrase.words:
                if word.is_emphasized:
                    primary_color = word.color
                    break
            
            # Professional TikTok-style font sizes
            font_size = 72 if has_emphasis else 56  # Clean, readable sizes
            duration = phrase.end_time - phrase.start_time
            
            # Ensure minimum duration for visibility (captions might be too short!)
            min_duration = 1.0  # At least 1 second
            if duration < min_duration:
                duration = min_duration
                logger.info(f"⏰ Extended caption duration from {phrase.end_time - phrase.start_time:.1f}s to {duration}s")
            
            # Add visual indicators for viral moments
            display_text = phrase.text
            if has_emphasis:
                display_text = f"🔥 {phrase.text.upper()} 🔥"
            
            # Create PROFESSIONAL viral text clip
            text_clip = TextClip(
                display_text,
                fontsize=font_size,
                color='white',  # Always white for best contrast
                stroke_color='black',
                stroke_width=3  # Clean outline
            ).set_duration(duration)
            
            # Skip glow effect for performance - just use thicker stroke for emphasis
            if has_emphasis:
                logger.info(f"🔥 Using viral styling for emphasized caption")
            
            # Better positioning for TikTok format - center in bottom third
            if video_size[1] > 1800:  # Vertical TikTok format
                # Position in the bottom third, but not too close to bottom
                bottom_third_start = video_size[1] * 2 // 3
                y_position = bottom_third_start + 100  # Center of bottom third
            else:
                y_position = video_size[1] - 80  # Closer to bottom for landscape
            
            text_clip = text_clip.with_position(('center', y_position))
            
            logger.info(f"✅ Caption created: '{display_text}' at y={y_position}, duration={duration:.1f}s")
            
            # Set timing
            text_clip = text_clip.with_start(phrase.start_time)
            
            logger.info(f"✨ Created caption: '{display_text}' ({duration:.1f}s, emphasized: {has_emphasis})")
            return text_clip
            
        except Exception as e:
            logger.error(f"Failed to create animated caption: {e}")
            return None
    
    def convert_to_vertical_format(self, video: object) -> object:
        """Convert video to TikTok-ready 9:16 vertical format"""
        if not MOVIEPY_AVAILABLE:
            return video
        
        try:
            original_w, original_h = video.size
            target_ratio = 9 / 16  # TikTok ratio
            current_ratio = original_w / original_h
            
            if abs(current_ratio - target_ratio) < 0.1:
                # Already close to vertical
                return video
            
            # Use STANDARD TikTok dimensions instead of calculated ones
            new_w = 1080  # Standard TikTok width
            new_h = 1920  # Standard TikTok height (9:16 ratio)
            
            # Create blurred background
            try:
                background = video.resized((new_w, new_h)).with_effects("blur", 10)
            except:
                # Fallback: simple resize without blur
                try:
                    background = video.resized((new_w, new_h))
                except:
                    background = video
            
            # Overlay original video in center
            final_video = CompositeVideoClip([
                background,
                video.with_position('center')
            ], size=(new_w, new_h))
            
            logger.info(f"✅ Converted to vertical format: {new_w}x{new_h}")
            return final_video
            
        except Exception as e:
            logger.error(f"Failed to convert to vertical format: {e}")
            return video
    
    def download_sound_effect(self, sound_name: str) -> Optional[Path]:
        """Download a sound effect"""
        try:
            if sound_name not in self.sound_effects:
                return None
            
            url = self.sound_effects[sound_name]
            sound_path = self.temp_dir / f"{sound_name}.wav"
            
            if sound_path.exists():
                return sound_path
            
            response = requests.get(url)
            response.raise_for_status()
            
            with open(sound_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✅ Downloaded sound effect: {sound_path}")
            return sound_path
            
        except Exception as e:
            logger.error(f"Error downloading sound effect: {e}")
            return None
    
    def enhance_clip(
        self, 
        video_path: Path, 
        text_overlay: str = None,
        sound_effect: str = None,
        text_position: str = "center",
        text_duration: float = 2.0,
        font_size: int = 50,
        use_captions: bool = True,
        vertical_format: bool = False
    ) -> Optional[Path]:
        """Enhance a clip with animated captions, AI text and sound effects"""
        
        if not MOVIEPY_AVAILABLE:
            logger.error("MoviePy not available. Install with: pip install moviepy")
            return None
        
        try:
            # Load the video
            video = VideoFileClip(str(video_path))
            clips_to_composite = [video]
            
            # Convert to vertical format if requested
            if vertical_format:
                video = self.convert_to_vertical_format(video)
                clips_to_composite = [video]
            
            # Try to add animated captions first
            caption_clips = []
            if use_captions:
                logger.info("🎤 Starting transcription process...")
                words = self.transcribe_audio(video_path)
                logger.info(f"📝 Transcription result: {len(words) if words else 0} words")
                
                if words:
                    logger.info(f"First few words: {[w.text for w in words[:5]]}")
                    
                    # Enhance words for virality
                    enhanced_words = self.enhance_words_for_virality(words)
                    emphasized_count = sum(1 for w in enhanced_words if w.is_emphasized)
                    logger.info(f"📈 Enhanced words: {emphasized_count} emphasized out of {len(enhanced_words)}")
                    
                    # Group into phrases
                    phrases = self.group_words_into_phrases(enhanced_words)
                    logger.info(f"📝 Created {len(phrases)} phrases for captions")
                    
                    # Create animated captions
                    for i, phrase in enumerate(phrases):
                        logger.info(f"🎬 Creating caption {i+1}/{len(phrases)}: '{phrase.text[:50]}...'")
                        caption_clip = self.create_animated_caption(phrase, video.size)
                        if caption_clip:
                            caption_clips.append(caption_clip)
                            logger.info(f"✅ Caption {i+1} created successfully")
                        else:
                            logger.warning(f"❌ Caption {i+1} creation failed")
                    
                    if caption_clips:
                        clips_to_composite.extend(caption_clips)
                        logger.info(f"🔥 ADDED {len(caption_clips)} VIRAL CAPTIONS TO VIDEO!")
                        logger.info(f"📊 Total clips to composite: {len(clips_to_composite)} (1 video + {len(caption_clips)} captions)")
                        
                        # Log each caption timing
                        for i, clip in enumerate(caption_clips):
                            start_time = getattr(clip, 'start', 'unknown')
                            duration = getattr(clip, 'duration', 'unknown')
                            logger.info(f"   Caption {i+1}: starts at {start_time}s, duration {duration}s")
                    else:
                        logger.error("❌ NO CAPTION CLIPS WERE GENERATED!")
                        # Create a VERY visible test caption as fallback
                        logger.info("🧪 Creating SUPER VISIBLE test caption as fallback...")
                        test_clip = TextClip(
                            "🔥🔥🔥 TEST CAPTION VISIBLE? 🔥🔥🔥",
                            fontsize=100,
                            color='red',
                            stroke_color='white',
                            stroke_width=6,
                            size=video.size
                        ).set_duration(min(10.0, video.duration)).with_position(('center', video.size[1] // 2))  # Center of screen
                        clips_to_composite.append(test_clip)
                        logger.info("✅ MASSIVE test caption added to center of screen")
                else:
                    logger.warning("No transcription available, falling back to static text")
                    use_captions = False
            
            # Fallback to static text overlay if captions failed or disabled
            if not use_captions and text_overlay:
                try:
                    # Create text clip with basic parameters for compatibility
                    text_clip = TextClip(
                        text_overlay,
                        fontsize=font_size,
                        color='white',
                        size=video.size  # Use video size
                    ).set_duration(text_duration)
                    
                    # Simple center positioning
                    clips_to_composite.append(text_clip)
                    logger.info(f"✅ Added static text overlay: {text_overlay}")
                    
                except Exception as e:
                    logger.warning(f"Could not add text overlay: {e}")
            
            # Composite the video
            final_video = CompositeVideoClip(clips_to_composite)
            
            # Add sound effect
            if sound_effect:
                sound_path = self.download_sound_effect(sound_effect)
                if sound_path and sound_path.exists():
                    try:
                        sound_clip = AudioFileClip(str(sound_path))
                        # Mix with original audio
                        final_audio = final_video.audio.volumex(0.7)  # Lower original volume
                        mixed_audio = final_audio.set_duration(final_video.duration)
                        
                        # Add sound effect at the beginning
                        if sound_clip.duration <= final_video.duration:
                            sound_clip = sound_clip.set_start(0.2)  # Slight delay
                            mixed_audio = CompositeAudioClip([mixed_audio, sound_clip])
                        
                        final_video = final_video.set_audio(mixed_audio)
                        sound_clip.close()
                    except Exception as e:
                        logger.warning(f"Could not add sound effect: {e}")
            
            # Export enhanced video
            suffix = "_captioned" if caption_clips else "_enhanced"
            if vertical_format:
                suffix += "_vertical"
            output_path = self.temp_dir / f"{suffix}_{video_path.stem}.mp4"
            
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac'
            )
            
            # Clean up
            video.close()
            final_video.close()
            for clip in caption_clips:
                if hasattr(clip, 'close'):
                    clip.close()
            
            logger.info(f"✅ Enhanced clip saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error enhancing clip: {e}")
            return None
    
    def auto_enhance_clip(self, clip_url: str, context: str = "", use_captions: bool = True) -> Optional[Dict]:
        """Automatically download and enhance a clip"""
        try:
            logger.info(f"🎬 Auto-enhancing clip: {clip_url}")
            
            # Download the clip
            video_path = self.download_clip(clip_url)
            if not video_path:
                return None
            
            # Generate AI text (fallback if captions fail)
            viral_text = self.generate_viral_text(context)
            
            # Determine sound effect based on context
            sound_effect = "airhorn"  # Default
            if any(word in context.lower() for word in ["chat", "spam", "messages"]):
                sound_effect = "wow"
            elif any(word in context.lower() for word in ["viewer", "spike", "growth"]):
                sound_effect = "epic"
            
            # Enhance the clip with captions
            enhanced_path = self.enhance_clip(
                video_path,
                text_overlay=viral_text,
                sound_effect=sound_effect,
                text_position="center",
                text_duration=2.5,
                font_size=60,
                use_captions=use_captions
            )
            
            if enhanced_path:
                return {
                    'original_url': clip_url,
                    'original_path': video_path,
                    'enhanced_path': enhanced_path,
                    'text_overlay': viral_text,
                    'sound_effect': sound_effect,
                    'has_captions': use_captions
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Auto-enhancement failed: {e}")
            return None
    
    def create_captioned_clip(self, clip_url: str, vertical: bool = True) -> Optional[Dict]:
        """Create a MrBeast-style captioned clip optimized for TikTok/Shorts"""
        try:
            logger.info(f"🎭 Creating captioned clip: {clip_url}")
            
            # Download the clip
            video_path = self.download_clip(clip_url)
            if not video_path:
                return None
            
            # Enhance with captions and vertical format
            enhanced_path = self.enhance_clip(
                video_path,
                use_captions=True,
                vertical_format=vertical,
                sound_effect="airhorn",  # Default viral sound
                font_size=75
            )
            
            if enhanced_path:
                return {
                    'original_url': clip_url,
                    'original_path': video_path,
                    'enhanced_path': enhanced_path,
                    'format': 'vertical' if vertical else 'original',
                    'has_captions': True,
                    'style': 'mrbeast'
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Caption clip creation failed: {e}")
            return None
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
            logger.info("🧹 Cleaned up temporary files")
        except Exception as e:
            logger.error(f"Error cleaning up: {e}")


def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    optional_missing = []
    
    if not MOVIEPY_AVAILABLE:
        missing.append("moviepy")
    
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("yt-dlp")
    
    if not WHISPER_AVAILABLE:
        optional_missing.append("openai-whisper")
    
    if not openai:
        optional_missing.append("openai")
    
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            if dep == "moviepy":
                print(f"   pip install {dep}")
            elif dep == "yt-dlp":
                print(f"   pip install {dep}")
        return False
    
    if optional_missing:
        print("⚠️  Missing optional dependencies (for enhanced features):")
        for dep in optional_missing:
            print(f"   pip install {dep}")
    
    return True


if __name__ == "__main__":
    # Quick test
    if check_dependencies():
        enhancer = ClipEnhancer()
        print("✅ Clip enhancer ready!")
    else:
        print("❌ Please install missing dependencies first")
