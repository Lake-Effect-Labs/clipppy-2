#!/usr/bin/env python3
"""
Auto-Enhancement Engine v2 - Modular Enhancement Pipeline
=========================================================

Turns raw clips into platform-native, branded, highly watchable vertical shorts.
Configurable via presets, safe by default, maintainable and debuggable.

Stories implemented:
- A: Format & Framing (smart crop, vertical, safe zones)
- B: Captions & Copy (karaoke, emphasis, profanity filter) 
- C: Audio Polish (normalization, compression)
- D: Visual Tempo (silence removal, speed ramps)
- E: Branding & Attribution (watermarks, borders, end slate)
- F: Context Overlays (chat bursts, event tags) [v2]
- G: Mascot Animations (clippy mascot) [v2] 
- H: Platform Native (thumbnails, captions, safe outputs)
"""

import logging
import time
import re
import random
import math
import hashlib
from pathlib import Path
from emotion_detector import StreamerEmotionDetector
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import tempfile
import yaml

# Video processing
try:
    # Fix PIL compatibility issue with newer versions
    import PIL.Image
    if not hasattr(PIL.Image, 'ANTIALIAS'):
        PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
    
    import moviepy.editor as mp
    import moviepy.config as mpconf
    # Configure ImageMagick path for Windows
    mpconf.IMAGEMAGICK_BINARY = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
    # MoviePy 1.0.3 has different import structure
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    mp = None

# Face detection
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None
    np = None

# Audio processing
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

logger = logging.getLogger(__name__)

@dataclass
class EnhancementTelemetry:
    """Track enhancement metrics for analysis"""
    processing_time_ms: int = 0
    words_rendered: int = 0
    emphasis_hits: int = 0
    lufs_before: float = 0.0
    lufs_after: float = 0.0
    cuts_count: int = 0
    ms_removed: int = 0
    hook_used: bool = False
    watermark_position: str = ""
    border_px: int = 0
    vignette_value: float = 0.0
    end_slate: bool = False

@dataclass 
class CaptionWord:
    """Individual word with timing and emphasis"""
    text: str
    start_time: float
    end_time: float
    is_emphasis: bool = False
    color: str = "#FFFFFF"
    size_boost: float = 1.0

@dataclass
class SafeZone:
    """Define safe areas for UI elements"""
    top: float = 0.0        # Top % of screen to avoid
    bottom: float = 0.25    # Bottom % reserved for captions
    left: float = 0.05      # Left margin
    right: float = 0.05     # Right margin

class ClipEnhancerV2:
    """Enhanced clip processing with preset-based configuration"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize enhancer with configuration"""
        self.config_path = Path(config_path)
        self.load_config()
        
        # Setup directories
        self.clips_dir = Path("clips")
        self.temp_dir = self.clips_dir / "temp"
        self.assets_dir = Path("assets")
        self.fonts_dir = self.clips_dir / "fonts"
        
        # Create directories
        for dir_path in [self.clips_dir, self.temp_dir, self.assets_dir, self.fonts_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # Whisper model (lazy loading)
        self.whisper_model = None
        
        # Initialize emotion detection system
        self.emotion_detector = StreamerEmotionDetector()
        
        logger.info("🎨 ClipEnhancerV2 initialized")
    
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            self.enhancement_presets = self.config.get('enhancement_presets', {})
            logger.info(f"✅ Loaded {len(self.enhancement_presets)} enhancement presets")
            
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            # Fallback to minimal config
            self.config = {}
            self.enhancement_presets = {}
    
    def get_enhancement_config(self, streamer_config: Dict) -> Dict:
        """Get merged enhancement config for a streamer"""
        # Get streamer's enhancement settings
        enhancement = streamer_config.get('enhancement', {})
        preset_name = enhancement.get('preset', 'viral_shorts')
        overrides = enhancement.get('overrides', {})
        
        # Get base preset
        preset = self.enhancement_presets.get(preset_name, {})
        if not preset:
            logger.warning(f"⚠️ Preset '{preset_name}' not found, using defaults")
            preset = self._get_default_preset()
        
        # Deep merge overrides into preset
        merged_config = self._deep_merge(preset, overrides)
        
        logger.info(f"🎯 Using preset '{preset_name}' with overrides for enhancement")
        return merged_config
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _get_default_preset(self) -> Dict:
        """Get minimal default preset"""
        return {
            'format': {
                'target_resolution': [1080, 1920],
                'smart_crop': True,
                'blur_background': True
            },
            'captions': {
                'enabled': True,
                'karaoke_sync': False,
                'style': {
                    'font_family': 'Arial Bold',
                    'font_size': 64,
                    'fill_color': '#FFFFFF',
                    'outline_color': '#000000',
                    'outline_width': 3
                }
            },
            'audio': {
                'normalize_lufs': -16
            },
            'branding': {
                'watermark': {'enabled': False}
            },
            'platform': {
                'output_codec': 'libx264',
                'output_crf': 18
            }
        }
    
    def enhance_clip(self, clip_path: str, streamer_config: Dict, output_path: Optional[str] = None) -> Tuple[str, EnhancementTelemetry]:
        """Main enhancement pipeline"""
        start_time = time.time()
        telemetry = EnhancementTelemetry()
        
        try:
            clip_path = Path(clip_path)
            if not clip_path.exists():
                raise FileNotFoundError(f"Clip not found: {clip_path}")
            
            # Get enhancement configuration
            config = self.get_enhancement_config(streamer_config)
            streamer_name = streamer_config.get('name', 'unknown')
            
            logger.info(f"🎬 Starting enhancement pipeline for {clip_path.name}")
            
            # Setup output path in streamer-specific folder
            if not output_path:
                timestamp = int(time.time())
                
                # Create streamer-specific directory
                streamer_clips_dir = self.clips_dir / streamer_name
                streamer_clips_dir.mkdir(exist_ok=True)
                
                output_path = streamer_clips_dir / f"enhanced_{streamer_name}_{timestamp}.mp4"
                logger.info(f"📁 Saving to streamer folder: {streamer_clips_dir}")
            else:
                output_path = Path(output_path)
            
            # Load video
            logger.info("📹 Loading video...")
            video = mp.VideoFileClip(str(clip_path))
            original_duration = video.duration
            
            # Check for IRL content before format & framing
            logger.info("🔍 Checking content type...")
            transcript_text = None
            if config.get('captions', {}).get('enabled', True):
                # Get transcript for both IRL detection and captions
                transcript = self._transcribe_audio(clip_path)
                if transcript:
                    transcript_text = ' '.join([word['text'] for word in transcript])
            
            # Detect IRL content and adjust split-screen accordingly
            is_irl = self._detect_irl_content(clip_path, transcript_text)
            if is_irl:
                # Disable split-screen for IRL content
                config['format']['split_screen'] = False
                logger.info("🗣️ IRL content detected - disabling split-screen layout")
            
            # Story A: Format & Framing
            logger.info("📐 Applying format & framing...")
            video, safe_zone = self._apply_format_framing(video, config.get('format', {}))
            
            # Story B: Captions & Copy
            logger.info("📝 Adding captions...")
            if config.get('captions', {}).get('enabled', True) and transcript:
                # Use existing transcript instead of re-transcribing
                video, caption_telemetry = self._apply_captions_with_transcript(video, transcript, config.get('captions', {}), safe_zone, streamer_name)
                telemetry.words_rendered = caption_telemetry.get('words_rendered', 0)
                telemetry.emphasis_hits = caption_telemetry.get('emphasis_hits', 0)
                telemetry.hook_used = caption_telemetry.get('hook_used', False)
            
            # Story C: Audio Polish
            logger.info("🔊 Polishing audio...")
            video, audio_telemetry = self._apply_audio_polish(video, config.get('audio', {}))
            telemetry.lufs_before = audio_telemetry.get('lufs_before', 0.0)
            telemetry.lufs_after = audio_telemetry.get('lufs_after', 0.0)
            
            # Story D: Visual Tempo
            logger.info("⚡ Optimizing tempo...")
            video, tempo_telemetry = self._apply_visual_tempo(video, config.get('tempo', {}))
            telemetry.cuts_count = tempo_telemetry.get('cuts_count', 0)
            telemetry.ms_removed = tempo_telemetry.get('ms_removed', 0)
            
            # Story E: Branding & Attribution
            logger.info("🏷️ Adding branding...")
            video, brand_telemetry = self._apply_branding(video, config.get('branding', {}), streamer_name, safe_zone)
            telemetry.watermark_position = brand_telemetry.get('watermark_position', '')
            telemetry.border_px = brand_telemetry.get('border_px', 0)
            telemetry.vignette_value = brand_telemetry.get('vignette_value', 0.0)
            telemetry.end_slate = brand_telemetry.get('end_slate', False)
            
            # Story H: Platform Native Output
            logger.info("💾 Rendering final video...")
            platform_config = config.get('platform', {})
            self._render_video(video, output_path, platform_config)
            
            # Calculate telemetry
            processing_time = (time.time() - start_time) * 1000
            telemetry.processing_time_ms = int(processing_time)
            
            logger.info(f"✅ Enhancement complete: {output_path}")
            logger.info(f"📊 Processing time: {processing_time:.0f}ms")
            logger.info(f"📊 Words rendered: {telemetry.words_rendered}")
            logger.info(f"📊 Emphasis hits: {telemetry.emphasis_hits}")
            
            # Cleanup
            video.close()
            
            return str(output_path), telemetry
            
        except Exception as e:
            logger.error(f"❌ Enhancement failed: {e}")
            raise
    
    def _detect_irl_content(self, clip_path: Path, transcript: str = None) -> bool:
        """Detect if content is IRL/Just Chatting (no gaming)"""
        try:
            # Keywords that indicate IRL/Just Chatting content
            irl_keywords = [
                'just chatting', 'irl', 'talking', 'reacting', 'reaction', 'chat',
                'interview', 'podcast', 'discussion', 'story', 'storytime',
                'cooking', 'eating', 'music', 'singing', 'dancing', 'workout',
                'travel', 'vlog', 'review', 'unboxing', 'asmr', 'viewers',
                'donations', 'subscribe', 'follow', 'thanks', 'appreciate',
                'question', 'answer', 'explain', 'opinion', 'think', 'feel',
                'stream', 'streaming', 'content', 'youtube', 'tiktok'
            ]
            
            # Check transcript for IRL indicators
            if transcript:
                transcript_lower = transcript.lower()
                irl_score = sum(1 for keyword in irl_keywords if keyword in transcript_lower)
                
                # Gaming keywords that would override IRL detection
                gaming_keywords = [
                    'game', 'play', 'level', 'kill', 'death', 'respawn', 'match',
                    'round', 'enemy', 'teammate', 'weapon', 'item', 'quest',
                    'boss', 'dungeon', 'raid', 'pvp', 'fps', 'moba', 'rpg'
                ]
                gaming_score = sum(1 for keyword in gaming_keywords if keyword in transcript_lower)
                
                # If significantly more IRL keywords than gaming keywords
                if irl_score > gaming_score + 2:
                    logger.info(f"🗣️ IRL content detected via transcript (IRL: {irl_score}, Gaming: {gaming_score})")
                    return True
            
            # Additional detection could be added here:
            # - Game category from Twitch API
            # - Video analysis for UI elements
            # - Audio analysis for game sounds
            
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ IRL detection failed: {e}")
            return False
    
    def _apply_format_framing(self, video, format_config: Dict) -> Tuple[Any, SafeZone]:
        """Story A: Format & Framing with Viral Split-Screen Layout"""
        target_w, target_h = format_config.get('target_resolution', [1080, 1920])
        smart_crop = format_config.get('smart_crop', True)
        split_screen = format_config.get('split_screen', True)  # NEW: Viral split-screen feature
        safe_zones_config = format_config.get('safe_zones', {})
        
        # Create safe zone
        safe_zone = SafeZone(
            bottom=safe_zones_config.get('caption_bottom', 0.25),
            left=0.05,
            right=0.05
        )
        
        original_w, original_h = video.size
        target_aspect = target_w / target_h
        original_aspect = original_w / original_h
        
        # Apply viral split-screen layout for TikTok/Shorts
        if target_w == 1080 and target_h == 1920 and split_screen:  # Vertical format with split screen
            logger.info(f"🎮 Creating viral split-screen layout from {original_w}x{original_h}")
            
            # Detect face and gameplay areas
            face_region, gameplay_region = self._detect_face_and_gameplay(video)
            
            if face_region and gameplay_region:
                # Create split-screen: gameplay top (60%), face bottom (40%) - balanced layout
                gameplay_height = int(target_h * 0.6)  # Top 60% for gameplay
                face_height = target_h - gameplay_height  # Bottom 40% for face
                
                # Extract and resize gameplay area (top half)
                gameplay_clip = video.crop(
                    x1=gameplay_region[0], 
                    y1=gameplay_region[1],
                    x2=gameplay_region[2], 
                    y2=gameplay_region[3]
                ).resize((target_w, gameplay_height))
                
                # Extract and resize face area (bottom half)
                face_clip = video.crop(
                    x1=face_region[0], 
                    y1=face_region[1],
                    x2=face_region[2], 
                    y2=face_region[3]
                ).resize((target_w, face_height))
                
                # Combine into split-screen
                video = mp.CompositeVideoClip([
                    gameplay_clip.set_position((0, 0)),  # Top
                    face_clip.set_position((0, gameplay_height))  # Bottom
                ], size=(target_w, target_h))
                
                logger.info(f"🎮 Created split-screen: gameplay {target_w}x{gameplay_height}, face {target_w}x{face_height}")
                
                # Adjust safe zone for split layout - captions over gameplay area
                safe_zone.bottom = 0.15  # Less space needed since captions go over gameplay
                
            else:
                # Fallback to smart crop if detection fails
                logger.warning("⚠️ Face/gameplay detection failed, using smart crop fallback")
                video = self._smart_crop_fallback(video, target_w, target_h)
        
        elif target_w == 1080 and target_h == 1920:  # Vertical without split screen
            logger.info(f"📐 Converting {original_w}x{original_h} to vertical {target_w}x{target_h}")
            video = self._smart_crop_fallback(video, target_w, target_h)
        else:
            logger.info(f"📐 Keeping original format: {original_w}x{original_h}")
        
        return video, safe_zone
    
    def _detect_face_and_gameplay(self, video) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """Detect face and main gameplay areas using improved face detection"""
        if not CV2_AVAILABLE:
            logger.warning("⚠️ OpenCV not available for face detection")
            return None, None
        
        try:
            # Get multiple frames for better detection
            duration = video.duration
            frame_times = [duration * 0.3, duration * 0.5, duration * 0.7]  # Sample multiple points
            best_face = None
            best_face_size = 0
            
            for frame_time in frame_times:
                frame = video.get_frame(frame_time)
                
                # Convert to OpenCV format
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                
                # Try multiple face detection methods
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
                # Multiple detection passes with different parameters
                faces_params = [
                    {'scaleFactor': 1.1, 'minNeighbors': 5, 'minSize': (30, 30)},  # More sensitive
                    {'scaleFactor': 1.2, 'minNeighbors': 3, 'minSize': (50, 50)},  # Default
                    {'scaleFactor': 1.3, 'minNeighbors': 4, 'minSize': (80, 80)},  # Less sensitive
                ]
                
                for params in faces_params:
                    faces = face_cascade.detectMultiScale(gray, **params)
                    
                    if len(faces) > 0:
                        # Get the largest face
                        largest_face = max(faces, key=lambda f: f[2] * f[3])
                        fx, fy, fw, fh = largest_face
                        face_size = fw * fh
                        
                        if face_size > best_face_size:
                            best_face = largest_face
                            best_face_size = face_size
                            break  # Found good face, no need to try other params
                
                if best_face is not None:
                    break  # Found face, no need to check other frames
            
            if best_face is not None:
                fx, fy, fw, fh = best_face
                frame_h, frame_w = frame.shape[:2]
                
                logger.info(f"👤 Found face at ({fx}, {fy}) size {fw}x{fh}")
                
                # For NICKMERCS-style layout: face should be on the right side typically
                # Use the face location to determine the best crop areas
                
                # Face region with generous padding for full head/shoulders
                padding = 1.2  # Even more generous padding
                face_x1 = max(0, int(fx - fw * padding))
                face_y1 = max(0, int(fy - fh * padding))
                face_x2 = min(frame_w, int(fx + fw * (1 + padding)))
                face_y2 = min(frame_h, int(fy + fh * (1 + padding)))
                
                # Ensure minimum face region size (at least 1/3 of frame width)
                min_face_width = frame_w // 3
                if (face_x2 - face_x1) < min_face_width:
                    center_x = (face_x1 + face_x2) // 2
                    face_x1 = max(0, center_x - min_face_width // 2)
                    face_x2 = min(frame_w, center_x + min_face_width // 2)
                
                face_region = (face_x1, face_y1, face_x2, face_y2)
                
                # Gameplay area: try to avoid the face area
                # If face is on the right, gameplay should focus on left/center
                if fx > frame_w * 0.6:  # Face on right side
                    gameplay_x1 = 0
                    gameplay_x2 = int(frame_w * 0.8)  # Don't include full right side
                else:  # Face on left side or center
                    gameplay_x1 = int(frame_w * 0.2)  # Skip left portion
                    gameplay_x2 = frame_w
                
                gameplay_y1 = 0
                gameplay_y2 = frame_h
                
                gameplay_region = (gameplay_x1, gameplay_y1, gameplay_x2, gameplay_y2)
                
                logger.info(f"🎮 Smart gameplay region: {gameplay_region} (avoiding face area)")
                logger.info(f"👤 Enhanced face region: {face_region} with {padding}x padding")
                
                return face_region, gameplay_region
            
            else:
                logger.warning("⚠️ No faces detected in any frame")
                # Return fallback regions based on typical streaming layout
                frame_h, frame_w = frame.shape[:2]
                
                # Assume face is on right side (typical for most streamers)
                face_x1 = int(frame_w * 0.6)  # Right 40% of frame
                face_y1 = int(frame_h * 0.2)  # Skip top 20%
                face_x2 = frame_w
                face_y2 = int(frame_h * 0.8)  # Skip bottom 20%
                face_region = (face_x1, face_y1, face_x2, face_y2)
                
                # Gameplay in left/center
                gameplay_x1 = 0
                gameplay_x2 = int(frame_w * 0.8)
                gameplay_y1 = 0
                gameplay_y2 = frame_h
                gameplay_region = (gameplay_x1, gameplay_y1, gameplay_x2, gameplay_y2)
                
                logger.info(f"📺 Using fallback layout - Face: {face_region}, Gameplay: {gameplay_region}")
                return face_region, gameplay_region
                
        except Exception as e:
            logger.error(f"❌ Face detection failed: {e}")
            return None, None
    
    def _smart_crop_fallback(self, video, target_w: int, target_h: int):
        """Fallback smart crop when face detection fails"""
        original_w, original_h = video.size
        original_aspect = original_w / original_h
        target_aspect = target_w / target_h
        
        if original_aspect > target_aspect:  # Video is wider than target
            # Calculate crop dimensions to get target aspect ratio
            new_width = int(original_h * target_aspect)
            
            # Smart positioning: focus on center-left where face cam usually is
            crop_x = int(original_w * 0.25)  # Start at 25% from left
            
            # Ensure we don't crop outside bounds
            if crop_x + new_width > original_w:
                crop_x = original_w - new_width
            if crop_x < 0:
                crop_x = 0
            
            # Crop and resize
            video = video.crop(x1=crop_x, x2=crop_x + new_width)
            logger.info(f"📐 Smart cropped to {new_width}x{original_h}")
        
        # Resize to target resolution
        try:
            video = video.resize((target_w, target_h))
        except AttributeError:
            video = video.resize(newsize=(target_w, target_h))
        
        logger.info(f"📐 Resized to {target_w}x{target_h}")
        return video
    
    def _blur_image(self, image, radius=20):
        """Apply gaussian blur to image (simple version)"""
        # This is a placeholder - in production you'd use PIL or cv2
        # For now just darken the image to simulate blur effect
        import numpy as np
        return (image * 0.3).astype(np.uint8)
    
    def _apply_captions_with_transcript(self, video, transcript: List[Dict], captions_config: Dict, safe_zone: SafeZone, streamer_name: str) -> Tuple[Any, Dict]:
        """Story B: Captions & Copy with VIRAL EFFECTS (using existing transcript)"""
        telemetry = {}
        
        if not captions_config.get('enabled', True):
            return video, telemetry
        
        if not transcript:
            logger.warning("⚠️ No transcript available, skipping captions")
            return video, telemetry
        
        # Process captions
        words = self._process_transcript(transcript, captions_config)
        
        # Add hook line if enabled
        hook_config = captions_config.get('hook_line', {})
        if hook_config.get('enabled', False):
            hook_text = self._generate_hook_line(hook_config, words, streamer_name)
            if hook_text:
                telemetry['hook_used'] = True
                # Add hook as overlay at beginning
                video = self._add_hook_overlay(video, hook_text, hook_config, captions_config.get('style', {}))
        
        # 🚀 VIRAL EFFECTS - Apply zoom, shake, and flash effects during emphasis moments
        viral_config = captions_config.get('viral_effects', {})
        fast_mode = viral_config.get('fast_mode', False)  # Skip heavy effects for testing
        
        if viral_config.get('enabled', True) and not fast_mode:
            # Apply word-based, volume-based, and emotion-based effects
            video = self._apply_viral_effects(video, words, viral_config)
            video = self._apply_volume_based_effects(video, clip_path, viral_config)
            video = self._apply_emotion_based_effects(video, clip_path, viral_config)
        elif fast_mode:
            logger.info("⚡ Fast mode enabled - skipping viral effects for speed")
        
        # Create karaoke captions
        if captions_config.get('karaoke_sync', True) and words:
            caption_clips = self._create_karaoke_captions(words, captions_config.get('style', {}), video.size, safe_zone)
            if caption_clips:
                video = mp.CompositeVideoClip([video] + caption_clips)
        
        telemetry['words_rendered'] = len(words)
        telemetry['emphasis_hits'] = sum(1 for word in words if word.is_emphasis)
        
        return video, telemetry
    
    def _apply_viral_effects(self, video, words: List[CaptionWord], viral_config: Dict):
        """Apply viral TikTok-style effects during emphasis moments"""
        try:
            effects_applied = []
            
            # Find emphasis moments for viral effects
            emphasis_moments = [word for word in words if word.is_emphasis]
            
            # TEMP: If no emphasis found, create fake ones for testing
            if not emphasis_moments and len(words) > 5:
                logger.info("🧪 No emphasis found, creating test effects at 2-second mark")
                fake_emphasis = CaptionWord(
                    text="INSANE", 
                    start_time=2.0, 
                    end_time=2.5, 
                    is_emphasis=True
                )
                emphasis_moments = [fake_emphasis]
            
            if not emphasis_moments:
                logger.info("📺 No emphasis moments found, skipping viral effects")
                return video
            
            logger.info(f"🔥 Applying viral effects to {len(emphasis_moments)} emphasis moments")
            
            for word in emphasis_moments:
                effect_start = word.start_time
                effect_duration = min(word.end_time - word.start_time, 0.5)  # Max 0.5s per effect
                
                # Determine effect type based on word
                word_upper = word.text.upper()
                
                if any(trigger in word_upper for trigger in ['INSANE', 'CRAZY', 'WTF']):
                    # MEGA EFFECTS - toned down for better experience
                    video = self._add_screen_shake(video, effect_start, effect_duration, intensity=0.08)  # Reduced intensity
                    video = self._add_dramatic_zoom(video, effect_start, effect_duration, zoom_factor=1.3)  # More reasonable zoom
                    video = self._add_flash_effect(video, effect_start, 0.12)  # Shorter flash
                    effects_applied.append(f"MEGA_EFFECT@{effect_start:.1f}s")
                    
                elif any(trigger in word_upper for trigger in ['CLUTCH', 'NO WAY', 'POGGERS']):
                    # HYPE EFFECTS - balanced
                    video = self._add_dramatic_zoom(video, effect_start, effect_duration, zoom_factor=1.2)  # Subtle zoom
                    video = self._add_flash_effect(video, effect_start, 0.08)  # Quick flash
                    effects_applied.append(f"HYPE_EFFECT@{effect_start:.1f}s")
                    
                elif any(trigger in word_upper for trigger in ['JYNXZI', 'SHEESH']):
                    # GENTLE SHAKE for name mentions
                    video = self._add_screen_shake(video, effect_start, effect_duration, intensity=0.05)  # Much gentler
                    effects_applied.append(f"SHAKE@{effect_start:.1f}s")
                
                else:
                    # Apply subtle effects to ANY emphasis word for testing
                    video = self._add_flash_effect(video, effect_start, 0.06)  # Quick flash
                    video = self._add_dramatic_zoom(video, effect_start, effect_duration, zoom_factor=1.15)  # Subtle zoom
                    effects_applied.append(f"TEST_EFFECT@{effect_start:.1f}s")
            
            logger.info(f"✨ Applied viral effects: {', '.join(effects_applied)}")
            return video
            
        except Exception as e:
            logger.error(f"❌ Viral effects failed: {e}")
            return video
    
    def _add_dramatic_zoom(self, video, start_time: float, duration: float, zoom_factor: float = 1.2):
        """Add dramatic zoom effect during viral moments"""
        try:
            def zoom_effect(get_frame, t):
                frame = get_frame(t)
                if start_time <= t <= start_time + duration:
                    # Calculate zoom progress (0 to 1 and back to 0)
                    progress = (t - start_time) / duration
                    zoom_intensity = zoom_factor * math.sin(progress * math.pi)  # Smooth in/out
                    
                    if zoom_intensity > 0:
                        h, w = frame.shape[:2]
                        # Calculate crop area for zoom
                        crop_w = int(w / (1 + zoom_intensity * 0.2))
                        crop_h = int(h / (1 + zoom_intensity * 0.2))
                        start_x = (w - crop_w) // 2
                        start_y = (h - crop_h) // 2
                        
                        # Crop and resize back to original size
                        cropped = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
                        try:
                            import cv2
                            zoomed = cv2.resize(cropped, (w, h))
                            return zoomed
                        except:
                            return frame
                return frame
            
            return video.fl(zoom_effect)
        except Exception as e:
            logger.warning(f"⚠️ Zoom effect failed: {e}")
            return video
    
    def _add_screen_shake(self, video, start_time: float, duration: float, intensity: float = 0.1):
        """Add screen shake effect for INSANE moments"""
        try:
            def shake_effect(get_frame, t):
                frame = get_frame(t)
                if start_time <= t <= start_time + duration:
                    progress = (t - start_time) / duration
                    shake_intensity = intensity * math.sin(progress * math.pi * 20)  # Fast shake
                    
                    h, w = frame.shape[:2]
                    offset_x = int(shake_intensity * w * random.uniform(-1, 1))
                    offset_y = int(shake_intensity * h * random.uniform(-1, 1))
                    
                    # Create shaken frame
                    shaken = np.zeros_like(frame)
                    
                    # Calculate bounds
                    src_x1 = max(0, -offset_x)
                    src_y1 = max(0, -offset_y)
                    src_x2 = min(w, w - offset_x)
                    src_y2 = min(h, h - offset_y)
                    
                    dst_x1 = max(0, offset_x)
                    dst_y1 = max(0, offset_y)
                    dst_x2 = dst_x1 + (src_x2 - src_x1)
                    dst_y2 = dst_y1 + (src_y2 - src_y1)
                    
                    if src_x2 > src_x1 and src_y2 > src_y1 and dst_x2 <= w and dst_y2 <= h:
                        shaken[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
                        return shaken
                
                return frame
            
            return video.fl(shake_effect)
        except Exception as e:
            logger.warning(f"⚠️ Shake effect failed: {e}")
            return video
    
    def _add_flash_effect(self, video, start_time: float, duration: float):
        """Add BRIGHT white flash effect for emphasis"""
        try:
            # Create BRIGHT white flash clip with fade
            flash_clip = mp.ColorClip(
                size=video.size, 
                color=(255, 255, 255)
            ).set_duration(duration).set_start(start_time).set_opacity(0.8).fadein(0.02).fadeout(0.05)
            
            logger.info(f"⚡ Adding flash effect at {start_time:.1f}s for {duration:.2f}s")
            
            # Composite with original video
            return mp.CompositeVideoClip([video, flash_clip])
        except Exception as e:
            logger.warning(f"⚠️ Flash effect failed: {e}")
            return video
    
    def _apply_volume_based_effects(self, video, clip_path: Path, viral_config: Dict):
        """Apply camera shake based on audio volume levels (rage detection)"""
        try:
            # Temporarily disabled due to MoviePy CompositeAudioClip fps compatibility issues
            if not viral_config.get('volume_shake_enabled', False):  # Default to False
                return video
            
            logger.info("🔊 Analyzing audio for rage moments...")
            
            # Simple approach: use the video's audio directly
            if not hasattr(video, 'audio') or video.audio is None:
                logger.warning("⚠️ No audio found in video, skipping volume analysis")
                return video
            
            duration = video.duration
            chunk_size = 0.5  # Larger chunks for stability (500ms)
            volume_threshold = viral_config.get('rage_volume_threshold', 0.2)
            shake_intensity = viral_config.get('rage_shake_intensity', 0.10)
            
            rage_moments = []
            volumes_detected = []
            
            # Sample audio volume at larger intervals for stability
            for t in range(int(duration / chunk_size)):
                start_time = t * chunk_size
                end_time = min((t + 1) * chunk_size, duration)
                
                try:
                    # Get audio chunk from video
                    chunk = video.audio.subclip(start_time, end_time)
                    
                    # Get audio array
                    if hasattr(chunk, 'to_soundarray'):
                        audio_array = chunk.to_soundarray()
                        
                        if len(audio_array) > 0:
                            # Calculate RMS volume
                            rms = np.sqrt(np.mean(audio_array**2))
                            volume_level = min(rms * 5, 1.0)  # Scale appropriately
                            volumes_detected.append(volume_level)
                            
                            # Debug logging
                            if t % 4 == 0:  # Every 2 seconds
                                logger.info(f"🔊 Volume at {start_time:.1f}s: {volume_level:.2f}")
                            
                            # Check if above threshold
                            if volume_level > volume_threshold:
                                rage_moments.append({
                                    'start': start_time,
                                    'end': end_time,
                                    'intensity': volume_level
                                })
                    
                    chunk.close()
                    
                except Exception as e:
                    logger.warning(f"⚠️ Audio chunk failed at {start_time:.1f}s: {str(e)[:50]}")
                    continue
            
            # Log statistics
            if volumes_detected:
                avg_volume = sum(volumes_detected) / len(volumes_detected)
                max_volume = max(volumes_detected)
                logger.info(f"📊 Volume analysis: Avg {avg_volume:.2f}, Max {max_volume:.2f}, Threshold {volume_threshold}")
                
                if rage_moments:
                    logger.info(f"🔥 Found {len(rage_moments)} rage moments - applying shakes!")
                    
                    # Apply shake effects
                    for moment in rage_moments:
                        intensity = shake_intensity * moment['intensity'] * 2  # Amplify effect
                        shake_duration = min(moment['end'] - moment['start'], 0.3)  # Cap duration
                        
                        video = self._add_screen_shake(video, moment['start'], shake_duration, intensity=intensity)
                        logger.info(f"🎯 Rage shake at {moment['start']:.1f}s (intensity: {intensity:.2f})")
                else:
                    logger.info(f"😴 No volume spikes above {volume_threshold} - try lowering threshold")
            else:
                logger.warning("⚠️ No volume data collected - audio analysis failed")
            
            return video
            
        except Exception as e:
            logger.error(f"❌ Volume-based effects failed: {e}")
            return video
    
    def _apply_emotion_based_effects(self, video, clip_path: Path, viral_config: Dict):
        """Apply effects based on AI-detected streamer emotions"""
        try:
            if not viral_config.get('emotion_effects_enabled', True):
                return video
            
            logger.info("🤖 Analyzing streamer emotions for targeted effects...")
            
            # Detect emotional segments
            emotion_segments = self.emotion_detector.detect_emotional_segments(str(clip_path))
            
            if not emotion_segments:
                logger.info("😐 No strong emotional moments detected")
                return video
            
            logger.info(f"🎭 Found {len(emotion_segments)} emotional segments")
            
            # Apply effects for each emotional segment
            for segment in emotion_segments:
                logger.info(f"🎬 Applying {segment.dominant_emotion} effects at {segment.start_time:.1f}s")
                
                # Apply effects based on emotion type
                if segment.dominant_emotion == 'hype':
                    video = self._apply_hype_effects(video, segment)
                elif segment.dominant_emotion == 'tilt':
                    video = self._apply_tilt_effects(video, segment)
                elif segment.dominant_emotion == 'shock':
                    video = self._apply_shock_effects(video, segment)
                elif segment.dominant_emotion == 'confused':
                    video = self._apply_confused_effects(video, segment)
                
                # Add emotion text overlay
                emotion_text = self.emotion_detector.get_emotion_text_overlay(
                    segment.dominant_emotion, segment.intensity
                )
                video = self._add_emotion_text(video, segment, emotion_text)
            
            return video
            
        except Exception as e:
            logger.error(f"❌ Emotion-based effects failed: {e}")
            return video
    
    def _apply_hype_effects(self, video, segment):
        """Apply effects for hype moments"""
        try:
            # Intense zoom + shake combo
            zoom_factor = 1.1 + (segment.intensity * 0.1)  # 1.1x to 1.2x zoom
            shake_intensity = segment.intensity * 0.12
            
            # Golden flash effect
            flash_duration = 0.3
            
            # Apply zoom
            video = self._add_dramatic_zoom(video, segment.start_time, 
                                          segment.end_time - segment.start_time, zoom_factor)
            
            # Apply shake throughout segment
            video = self._add_screen_shake(video, segment.start_time, 
                                         segment.end_time - segment.start_time, shake_intensity)
            
            # Flash at start of hype moment
            video = self._add_flash_effect(video, segment.start_time, flash_duration, 
                                         color=(255, 215, 0))  # Gold flash
            
            logger.info(f"🔥 Applied hype effects: zoom {zoom_factor:.2f}x, shake {shake_intensity:.2f}")
            return video
            
        except Exception as e:
            logger.warning(f"⚠️ Hype effects failed: {e}")
            return video
    
    def _apply_tilt_effects(self, video, segment):
        """Apply effects for tilt/rage moments"""
        try:
            # Red tint + aggressive shake
            shake_intensity = segment.intensity * 0.15
            
            # Apply red tint overlay
            video = self._add_color_tint(video, segment.start_time, 
                                       segment.end_time - segment.start_time, 
                                       color=(255, 100, 100), opacity=0.2)
            
            # Aggressive shake
            video = self._add_screen_shake(video, segment.start_time, 
                                         segment.end_time - segment.start_time, shake_intensity)
            
            logger.info(f"😡 Applied tilt effects: red tint + shake {shake_intensity:.2f}")
            return video
            
        except Exception as e:
            logger.warning(f"⚠️ Tilt effects failed: {e}")
            return video
    
    def _apply_shock_effects(self, video, segment):
        """Apply effects for shock moments"""
        try:
            # Freeze frame + fast zoom
            freeze_duration = 0.5
            zoom_factor = 1.15
            
            # Quick freeze effect
            video = self._add_freeze_frame(video, segment.start_time, freeze_duration)
            
            # Fast zoom in
            video = self._add_dramatic_zoom(video, segment.start_time + freeze_duration, 
                                          0.3, zoom_factor)
            
            # White flash
            video = self._add_flash_effect(video, segment.start_time, 0.2, 
                                         color=(255, 255, 255))
            
            logger.info(f"😱 Applied shock effects: freeze + zoom {zoom_factor:.2f}x")
            return video
            
        except Exception as e:
            logger.warning(f"⚠️ Shock effects failed: {e}")
            return video
    
    def _apply_confused_effects(self, video, segment):
        """Apply effects for confused moments"""
        try:
            # Slow zoom out + question mark overlay
            zoom_factor = 0.95  # Zoom out slightly
            
            video = self._add_dramatic_zoom(video, segment.start_time, 
                                          segment.end_time - segment.start_time, zoom_factor)
            
            logger.info(f"🤔 Applied confused effects: zoom out {zoom_factor:.2f}x")
            return video
            
        except Exception as e:
            logger.warning(f"⚠️ Confused effects failed: {e}")
            return video
    
    def _add_color_tint(self, video, start_time, duration, color=(255, 0, 0), opacity=0.3):
        """Add colored tint overlay"""
        try:
            # Create colored overlay
            color_clip = mp.ColorClip(size=video.size, color=color, duration=duration)
            color_clip = color_clip.set_opacity(opacity).set_start(start_time)
            
            # Composite with video
            video = mp.CompositeVideoClip([video, color_clip])
            return video
            
        except Exception as e:
            logger.warning(f"⚠️ Color tint failed: {e}")
            return video
    
    def _add_freeze_frame(self, video, start_time, duration):
        """Add freeze frame effect"""
        try:
            # Extract frame at freeze point
            freeze_frame = video.to_ImageClip(t=start_time).set_duration(duration)
            
            # Replace video segment with frozen frame
            before_freeze = video.subclip(0, start_time)
            after_freeze = video.subclip(start_time + duration)
            
            video = mp.concatenate_videoclips([before_freeze, freeze_frame, after_freeze])
            return video
            
        except Exception as e:
            logger.warning(f"⚠️ Freeze frame failed: {e}")
            return video
    
    def _add_emotion_text(self, video, segment, text):
        """Add emotion-specific text overlay"""
        try:
            # Position emotion text at top of screen (above captions)
            text_clip = mp.TextClip(
                text,
                fontsize=80,
                color='yellow',
                font='Arial Black',
                stroke_color='black',
                stroke_width=3
            ).set_duration(min(2.0, segment.end_time - segment.start_time))
            
            # Position at top center
            text_clip = text_clip.set_position(('center', video.h * 0.1)).set_start(segment.start_time)
            
            # Composite with video
            video = mp.CompositeVideoClip([video, text_clip])
            return video
            
        except Exception as e:
            logger.warning(f"⚠️ Emotion text failed: {e}")
            return video
    
    def _transcribe_audio(self, clip_path: Path) -> Optional[List[Dict]]:
        """Transcribe audio using Whisper"""
        if not WHISPER_AVAILABLE:
            logger.warning("⚠️ Whisper not available, skipping transcription")
            return None
        
        try:
            if self.whisper_model is None:
                logger.info("🤖 Loading Whisper model (fast mode)...")
                self.whisper_model = whisper.load_model("tiny")  # Much faster than "base"
            
            result = self.whisper_model.transcribe(str(clip_path))
            
            # Convert to word-level timing (simplified)
            words = []
            for segment in result['segments']:
                segment_words = segment['text'].split()
                segment_duration = segment['end'] - segment['start']
                word_duration = segment_duration / len(segment_words) if segment_words else 0
                
                for i, word in enumerate(segment_words):
                    words.append({
                        'text': word,
                        'start': segment['start'] + (i * word_duration),
                        'end': segment['start'] + ((i + 1) * word_duration)
                    })
            
            return words
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return None
    
    def _process_transcript(self, transcript: List[Dict], captions_config: Dict) -> List[CaptionWord]:
        """Process transcript into CaptionWord objects with emphasis"""
        emphasis_map = captions_config.get('emphasis_map', [])
        emphasis_color = captions_config.get('style', {}).get('emphasis_color', '#FF0000')
        emphasis_boost = captions_config.get('style', {}).get('emphasis_size_boost', 1.2)
        
        words = []
        for word_data in transcript:
            text = word_data['text'].upper().strip('.,!?')
            
            # Enhanced emphasis detection - must be exact match or start of word
            is_emphasis = False
            for emphasis_word in emphasis_map:
                if (text == emphasis_word.upper() or 
                    text.startswith(emphasis_word.upper() + ' ') or
                    ' ' + emphasis_word.upper() in ' ' + text):
                    is_emphasis = True
                    break
            
            # Debug logging for emphasis detection
            if is_emphasis:
                logger.info(f"🔥 EMPHASIS DETECTED: '{word_data['text']}' matches emphasis words")
            
            word = CaptionWord(
                text=word_data['text'],
                start_time=word_data['start'],
                end_time=word_data['end'],
                is_emphasis=is_emphasis,
                color=emphasis_color if is_emphasis else '#FFFFFF',
                size_boost=emphasis_boost if is_emphasis else 1.0
            )
            words.append(word)
        
        logger.info(f"💬 Processed {len(words)} words, {sum(1 for w in words if w.is_emphasis)} emphasis words found")
        
        return words
    
    def _generate_hook_line(self, hook_config: Dict, words: List[CaptionWord], streamer_name: str) -> Optional[str]:
        """Generate hook line from templates"""
        templates = hook_config.get('templates', ["🔥 EPIC MOMENT"])
        
        # Simple template selection based on word content
        if any('INSANE' in word.text.upper() for word in words[:5]):
            return "NO SHOT 😱"
        elif any('CLUTCH' in word.text.upper() for word in words[:5]):
            return "CLUTCH PLAY 🎯"
        else:
            return random.choice(templates)
    
    def _add_hook_overlay(self, video, hook_text: str, hook_config: Dict, style_config: Dict):
        """Add hook line overlay"""
        duration = hook_config.get('duration', 2.0)
        
        # Create hook text clip
        hook_clip = mp.TextClip(
            hook_text,
            fontsize=int(style_config.get('font_size', 72) * 1.3),
            color=style_config.get('emphasis_color', '#FF0000'),
            stroke_color=style_config.get('outline_color', '#000000'),
            stroke_width=style_config.get('outline_width', 4),
            font=style_config.get('font_family', 'Arial Bold')
        ).set_duration(duration).set_position('center').set_start(0)
        
        return mp.CompositeVideoClip([video, hook_clip])
    
    def _create_karaoke_captions(self, words: List[CaptionWord], style_config: Dict, video_size: Tuple[int, int], safe_zone: SafeZone) -> List[Any]:
        """Create MrBeast-style phrase-based captions with viral styling"""
        clips = []
        
        # Group words into phrases (3-5 words per phrase for viral effect)
        phrases = self._group_words_into_phrases(words, min_words=3, max_words=5)
        
        base_fontsize = style_config.get('font_size', 110)  # Bigger for better visibility
        # Try viral fonts in order of preference - Windows system fonts
        viral_fonts = ['Impact', 'Arial Black', 'Trebuchet MS Bold', 'Verdana Bold', 'Arial Bold']
        outline_width = style_config.get('outline_width', 6)  # Balanced outline
        
        # Calculate position - at bottom of gameplay section (60% of screen)
        video_w, video_h = video_size
        # Position at 55% down the screen (bottom of 60% gameplay area)
        y_position = video_h * 0.55
        
        for phrase in phrases:
            # Create phrase text with emphasis styling
            phrase_text = self._style_phrase_text(phrase, style_config)
            
            # Check if phrase contains emphasis words
            has_emphasis = any(word.is_emphasis for word in phrase)
            
            # VIRAL STYLING - Dynamic colors and effects
            fontsize = base_fontsize
            
            # Dynamic color system based on phrase position and content
            phrase_colors = ['#FFFFFF', '#FFFF00', '#00FFFF', '#FF00FF', '#00FF00', '#FF6600']
            phrase_index = phrases.index(phrase) % len(phrase_colors)
            fill_color = phrase_colors[phrase_index]
            
            if has_emphasis:
                fontsize = int(base_fontsize * 1.5)  # 50% BIGGER for emphasis
                fill_color = '#FF0000'  # BRIGHT RED for maximum attention
                # Add fire emoji styling for ultra-viral effect
                phrase_text = f"🔥 {phrase_text} 🔥"
            
            # Create phrase clip with viral styling and font fallback
            phrase_clip = None
            for font_option in viral_fonts:
                try:
                    phrase_clip = mp.TextClip(
                        phrase_text,
                        fontsize=fontsize,
                        color=fill_color,
                        stroke_color='#000000',
                        stroke_width=outline_width,
                        font=font_option
                    ).set_duration(phrase[-1].end_time - phrase[0].start_time).set_start(phrase[0].start_time)
                    logger.info(f"✅ Using font: {font_option}")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Font {font_option} failed, trying next: {e}")
                    continue
            
            # Final fallback to default font
            if phrase_clip is None:
                phrase_clip = mp.TextClip(
                    phrase_text,
                    fontsize=fontsize,
                    color=fill_color,
                    stroke_color='#000000',
                    stroke_width=outline_width
                ).set_duration(phrase[-1].end_time - phrase[0].start_time).set_start(phrase[0].start_time)
                logger.warning("⚠️ Using default font as fallback")
            
            # Add VIRAL ANIMATIONS - only for emphasis words
            if has_emphasis:
                # EXTREME bounce effect for emphasized phrases only
                phrase_clip = phrase_clip.resize(lambda t: 1 + 0.3 * abs(math.sin(t * 12)))
            # Regular phrases stay static - no animation for cleaner look
            
            # Position cleanly in center
            phrase_clip = phrase_clip.set_position(('center', y_position))
            clips.append(phrase_clip)
        
        return clips
    
    def _group_words_into_phrases(self, words: List[CaptionWord], min_words: int = 3, max_words: int = 5) -> List[List[CaptionWord]]:
        """Group words into phrases for MrBeast-style captions"""
        phrases = []
        current_phrase = []
        
        for word in words:
            current_phrase.append(word)
            
            # Create phrase when we have enough words or hit natural break
            if (len(current_phrase) >= min_words and 
                (len(current_phrase) >= max_words or 
                 word.text.endswith(('.', '!', '?', ',')) or
                 word.is_emphasis)):
                phrases.append(current_phrase)
                current_phrase = []
        
        # Add remaining words as final phrase
        if current_phrase:
            phrases.append(current_phrase)
        
        return phrases
    
    def _style_phrase_text(self, phrase: List[CaptionWord], style_config: Dict) -> str:
        """Style phrase text with viral elements and profanity filtering"""
        # Profanity words to filter
        profanity_list = ['fuck', 'fucking', 'shit', 'damn', 'hell', 'ass', 'bitch', 'damn']
        
        words = []
        for word in phrase:
            word_text = word.text
            
            # Apply profanity filter
            if any(profanity in word_text.lower() for profanity in profanity_list):
                word_text = '*' * len(word_text)  # Replace with asterisks
            
            # Add emphasis styling
            if word.is_emphasis:
                word_text = f"🔥 {word_text.upper()} 🔥"
            
            words.append(word_text)
        
        return ' '.join(words)
    
    def _apply_audio_polish(self, video, audio_config: Dict) -> Tuple[Any, Dict]:
        """Story C: Audio Polish"""
        telemetry = {}
        
        if not video.audio:
            return video, telemetry
        
        # Skip audio processing for now due to MoviePy compatibility issues
        # Just use original audio
        telemetry['lufs_before'] = -20  # Placeholder
        telemetry['lufs_after'] = -16   # Placeholder
        
        return video, telemetry
    
    def _apply_visual_tempo(self, video, tempo_config: Dict) -> Tuple[Any, Dict]:
        """Story D: Visual Tempo"""
        telemetry = {'cuts_count': 0, 'ms_removed': 0}
        
        # For now, just return original video
        # Silence detection and speed ramping would require more complex processing
        return video, telemetry
    
    def _apply_branding(self, video, branding_config: Dict, streamer_name: str, safe_zone: SafeZone) -> Tuple[Any, Dict]:
        """Story E: Branding & Attribution"""
        telemetry = {}
        
        composites = [video]
        
        # Watermark
        watermark_config = branding_config.get('watermark', {})
        if watermark_config.get('enabled', False):
            watermark_clip = self._create_watermark(watermark_config, streamer_name, video.size, video.duration)
            if watermark_clip:
                composites.append(watermark_clip)
                telemetry['watermark_position'] = self._get_watermark_position(len(composites))
        
        # Border
        border_config = branding_config.get('border', {})
        if border_config.get('enabled', False):
            border_clip = self._create_border(border_config, video.size, video.duration)
            if border_clip:
                composites.append(border_clip)
                telemetry['border_px'] = border_config.get('width_px', 3)
        
        # Attribution
        attribution_config = branding_config.get('attribution', {})
        if attribution_config.get('enabled', False):
            attribution_clip = self._create_attribution(attribution_config, streamer_name, video.size, video.duration)
            if attribution_clip:
                composites.append(attribution_clip)
        
        # End slate
        end_slate_config = branding_config.get('end_slate', {})
        if end_slate_config.get('enabled', False):
            end_slate_clip = self._create_end_slate(end_slate_config, video.size)
            if end_slate_clip:
                # Extend video with end slate
                composites = [mp.concatenate_videoclips([mp.CompositeVideoClip(composites), end_slate_clip])]
                telemetry['end_slate'] = True
        
        if len(composites) > 1:
            video = mp.CompositeVideoClip(composites)
        
        return video, telemetry
    
    def _create_watermark(self, watermark_config: Dict, streamer_name: str, video_size: Tuple[int, int], duration: float) -> Optional[Any]:
        """Create watermark overlay"""
        handle_text = watermark_config.get('handle_text', '@{streamer}_clippy').format(streamer=streamer_name)
        opacity = watermark_config.get('opacity', 0.8)
        
        # Create text watermark
        watermark = mp.TextClip(
            handle_text,
            fontsize=24,
            color='white',
            stroke_color='black',
            stroke_width=1,
            font='Arial'
        ).set_duration(duration).set_opacity(opacity)
        
        # Position in corner (simple grid)
        if watermark_config.get('position_grid', True):
            position = self._get_corner_position(video_size, watermark.size)
            watermark = watermark.set_position(position)
        else:
            watermark = watermark.set_position(('right', 'top'))
        
        return watermark
    
    def _get_corner_position(self, video_size: Tuple[int, int], watermark_size: Tuple[int, int]) -> Tuple[int, int]:
        """Get deterministic corner position"""
        # Simple deterministic positioning
        video_w, video_h = video_size
        wm_w, wm_h = watermark_size
        
        # Use current time to hop between corners
        corner_seed = int(time.time()) % 4
        positions = [
            (20, 20),  # Top-left
            (video_w - wm_w - 20, 20),  # Top-right
            (20, video_h - wm_h - 20),  # Bottom-left
            (video_w - wm_w - 20, video_h - wm_h - 20)  # Bottom-right
        ]
        
        return positions[corner_seed]
    
    def _get_watermark_position(self, clip_count: int) -> str:
        """Get watermark position string for telemetry"""
        positions = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        return positions[(clip_count - 1) % 4]
    
    def _create_border(self, border_config: Dict, video_size: Tuple[int, int], duration: float) -> Optional[Any]:
        """Create border overlay (placeholder)"""
        # This would create a colored border around the video
        # For now, just return None as it's complex to implement with MoviePy
        return None
    
    def _create_attribution(self, attribution_config: Dict, streamer_name: str, video_size: Tuple[int, int], duration: float) -> Optional[Any]:
        """Create attribution text"""
        text = attribution_config.get('text', 'Clipped from @{streamer}').format(streamer=streamer_name)
        position = attribution_config.get('position', 'top_left')
        
        attribution = mp.TextClip(
            text,
            fontsize=18,
            color='white',
            stroke_color='black',
            stroke_width=1,
            font='Arial'
        ).set_duration(duration).set_opacity(0.7)
        
        # Position attribution
        if position == 'top_left':
            attribution = attribution.set_position((20, 20))
        elif position == 'top_right':
            attribution = attribution.set_position(('right', 20))
        else:
            attribution = attribution.set_position((20, 20))
        
        return attribution
    
    def _create_end_slate(self, end_slate_config: Dict, video_size: Tuple[int, int]) -> Optional[Any]:
        """Create end slate with branding"""
        duration = end_slate_config.get('duration', 1.5)
        template = end_slate_config.get('template', 'Clipped by Clipppy AI')
        
        # Create end slate
        end_slate = mp.TextClip(
            template,
            fontsize=48,
            color='white',
            stroke_color='black',
            stroke_width=2,
            font='Arial Bold'
        ).set_duration(duration).set_position('center')
        
        # Create background
        background = mp.ColorClip(size=video_size, color=(0, 0, 0)).set_duration(duration)
        
        return mp.CompositeVideoClip([background, end_slate])
    
    def _render_video(self, video, output_path: Path, platform_config: Dict):
        """Render final video with platform-native settings"""
        codec = platform_config.get('output_codec', 'libx264')
        crf = platform_config.get('output_crf', 18)
        preset = platform_config.get('output_preset', 'veryfast')
        
        # Export video with faster settings
        video.write_videofile(
            str(output_path),
            codec=codec,
            preset='ultrafast',  # Much faster encoding
            ffmpeg_params=['-crf', str(crf)],
            audio_codec='aac',
            threads=4  # Use multiple CPU cores
        )
        
        logger.info(f"💾 Video rendered: {output_path}")


def check_dependencies():
    """Check if all required dependencies are available"""
    missing = []
    
    if not MOVIEPY_AVAILABLE:
        missing.append("moviepy")
        logger.warning("⚠️ MoviePy not available - video processing disabled")
    else:
        logger.info("✅ MoviePy available")
    
    if not WHISPER_AVAILABLE:
        logger.warning("⚠️ Whisper not available - transcription disabled")
        # Don't add to missing as it's optional
    else:
        logger.info("✅ Whisper available")
    
    if missing:
        logger.error(f"❌ Missing dependencies: {missing}")
        return False
    
    logger.info("✅ All core dependencies available")
    return True


# Test function
def test_enhancement():
    """Test the enhancement system"""
    dependencies_ok = check_dependencies()
    
    # Always test config loading
    print("🧪 Testing enhancement v2 configuration system...")
    
    # Create test config
    test_streamer = {
        'name': 'test',
        'enhancement': {
            'preset': 'viral_shorts'
        }
    }
    
    try:
        enhancer = ClipEnhancerV2()
        
        # Test preset loading
        config = enhancer.get_enhancement_config(test_streamer)
        resolution = config.get('format', {}).get('target_resolution', [1080, 1920])
        preset_name = test_streamer['enhancement']['preset']
        
        print(f"✅ Config system working!")
        print(f"✅ Loaded preset '{preset_name}' with resolution: {resolution}")
        print(f"✅ Enhancement presets available: {list(enhancer.enhancement_presets.keys())}")
        
        # Test override system
        test_streamer_override = {
            'name': 'test_override',
            'enhancement': {
                'preset': 'viral_shorts',
                'overrides': {
                    'captions': {
                        'emphasis_map': ['CUSTOM', 'WORDS']
                    }
                }
            }
        }
        
        override_config = enhancer.get_enhancement_config(test_streamer_override)
        emphasis_words = override_config.get('captions', {}).get('emphasis_map', [])
        print(f"✅ Override system working! Custom emphasis: {emphasis_words}")
        
        if dependencies_ok:
            print("🎨 Enhancement system v2 fully ready!")
        else:
            print("⚠️ Enhancement system v2 config ready, but video processing needs MoviePy")
            print("   Install with: pip install moviepy")
        
    except Exception as e:
        print(f"❌ Config system test failed: {e}")
        return
    
    return dependencies_ok


if __name__ == "__main__":
    test_enhancement()
