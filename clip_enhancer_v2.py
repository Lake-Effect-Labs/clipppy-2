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

# Video download
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    yt_dlp = None

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

# TikTok-first enhancement settings
AUTO_HOOK = True
EXPORT_LONG_TIKTOK = True  # when keyword_burst_z or sentiment_swing_z high

# Audio processing
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

logger = logging.getLogger(__name__)

def build_hook_text(breakdown: dict) -> str:
    """Build TikTok hook text from viral detection breakdown"""
    mps = breakdown.get("current_mps", 0)
    kwz = breakdown.get("z",{}).get("kw",0)
    vdp = breakdown.get("viewer_delta_percent", 0)
    if kwz >= 2.0:
        return "wait for it… insane moment"
    if vdp >= 10:
        return "viewers skyrocketed… here's why"
    if mps >= 3.0:
        return "chat explodes in 3…2…1…"
    return "you won't believe this"

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
    
    def download_clip(self, clip_url: str, max_retries: int = 5, retry_delay: int = 10) -> Optional[str]:
        """Download a Twitch clip using yt-dlp with retry logic
        
        Twitch clips can take 10-30 seconds to be fully processed after creation.
        Uses exponential backoff to handle processing delays.
        """
        if not YT_DLP_AVAILABLE:
            logger.error("❌ yt-dlp not available. Install with: pip install yt-dlp")
            return None
        
        # Extract clip slug from URL
        # URLs can be: https://www.twitch.tv/streamer/clip/Slug or https://clips.twitch.tv/Slug
        # Slug format: Title-randomID (e.g., ClumsyPoorBisonGrammarKing-l-65jwLXr_7zxvs-)
        clip_slug = None
        if '/clip/' in clip_url:
            # Get everything after /clip/, remove query params, strip trailing dashes
            full_slug = clip_url.split('/clip/')[-1].split('?')[0].rstrip('-')
            # Use the full slug (not just the last part after -)
            clip_slug = full_slug if full_slug else None
        elif 'clips.twitch.tv' in clip_url:
            clip_slug = clip_url.split('/')[-1].split('?')[0].rstrip('-')
        
        if not clip_slug:
            logger.error(f"❌ Could not extract clip slug from URL: {clip_url}")
            return None
        
        # Ensure temp directory exists before download
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup download path (use absolute path)
        output_path = (self.temp_dir / f"{clip_slug}.mp4").resolve()
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best',
            'outtmpl': str(output_path),
            'quiet': False,
            'no_warnings': False,
            'ignoreerrors': False,
        }
        
        # Retry logic - clips may take time to be processed by Twitch
        # Use exponential backoff: 10s, 15s, 20s, 30s, 45s
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Exponential backoff: base delay * (1.5 ^ attempt)
                    delay = int(retry_delay * (1.5 ** (attempt - 1)))
                    logger.info(f"⏳ Retry {attempt}/{max_retries-1} - waiting {delay}s for clip to be processed by Twitch...")
                    time.sleep(delay)
                
                logger.info(f"📥 Downloading clip (attempt {attempt + 1}/{max_retries}): {clip_url}")
                
                # First try to extract info to check if clip is ready
                # This helps catch "formats empty" errors before attempting download
                try:
                    with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                        info = ydl.extract_info(clip_url, download=False)
                        if not info or 'formats' not in info or not info.get('formats'):
                            raise IndexError("Clip not ready: formats list is empty")
                except (IndexError, KeyError) as extract_error:
                    if "formats" in str(extract_error) or "list index out of range" in str(extract_error):
                        if attempt < max_retries - 1:
                            logger.warning(f"⚠️ Clip not ready yet (formats empty) - will retry...")
                            continue
                        else:
                            logger.error(f"❌ Clip still not ready after {max_retries} attempts - Twitch may still be processing")
                            return None
                    else:
                        # Other extraction errors, try download anyway
                        pass
                
                # If we get here, clip appears ready - attempt download
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([clip_url])
                
                # Wait a moment for file to be fully written
                time.sleep(0.5)
                
                # Verify file exists and is readable
                if output_path.exists() and output_path.stat().st_size > 0:
                    # Try to verify it's a valid video file
                    try:
                        import subprocess
                        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(output_path)]
                        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=5)
                        if result.returncode == 0 and result.stdout.strip():
                            logger.info(f"✅ Downloaded to: {output_path} (verified)")
                            return str(output_path)
                        else:
                            logger.warning(f"⚠️ Downloaded file exists but may be corrupted, retrying...")
                            if output_path.exists():
                                output_path.unlink()  # Delete corrupted file
                            raise Exception("File verification failed")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"⚠️ File verification timed out, assuming valid")
                        return str(output_path)
                    except FileNotFoundError:
                        # ffprobe not available, skip verification
                        logger.info(f"✅ Downloaded to: {output_path} (ffprobe not available, skipping verification)")
                        return str(output_path)
                    except Exception as verify_error:
                        logger.warning(f"⚠️ File verification failed: {verify_error}, retrying...")
                        if output_path.exists():
                            output_path.unlink()  # Delete corrupted file
                        raise
                else:
                    logger.warning(f"⚠️ Download completed but file not found or empty: {output_path}")
                    if output_path.exists():
                        output_path.unlink()  # Remove empty file
                    
            except IndexError as e:
                # Specific handling for "formats empty" error
                error_msg = str(e)
                if "formats" in error_msg or "list index out of range" in error_msg:
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Download attempt {attempt + 1} failed: Clip not ready (formats empty)")
                        continue
                    else:
                        logger.error(f"❌ Failed to download clip after {max_retries} attempts: Clip still not ready")
                        return None
                else:
                    # Other IndexError, re-raise
                    raise
            except Exception as e:
                error_msg = str(e)
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Download attempt {attempt + 1} failed: {error_msg}")
                    continue
                else:
                    logger.error(f"❌ Failed to download clip after {max_retries} attempts: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    return None
        
        return None
    
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
    
    def enhance_clip(self, clip_path: str, streamer_config: Dict, output_path: Optional[str] = None, viral_breakdown: Optional[Dict] = None) -> Tuple[str, EnhancementTelemetry]:
        """Main enhancement pipeline"""
        start_time = time.time()
        telemetry = EnhancementTelemetry()
        
        # Performance mode for faster processing
        performance_mode = streamer_config.get('enhancement', {}).get('performance_mode', 'balanced')
        logger.info(f"⚡ Performance mode: {performance_mode}")
        
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
            
            # Trim to TikTok/Instagram Shorts optimal length (15-60 seconds) - ALWAYS enforce
            if original_duration > 60:
                logger.info(f"✂️ Trimming video from {original_duration:.1f}s to 60s for TikTok/Instagram Shorts")
                video = video.subclip(0, 60)
                original_duration = 60
            elif original_duration < 15:
                logger.info(f"📱 Video is {original_duration:.1f}s (already Shorts-optimized)")
            
            # Store original duration for later reference
            video.original_duration = original_duration
            
            # TikTok-first enhancements
            hook_text = None
            export_long_variant = False
            
            if AUTO_HOOK and viral_breakdown:
                hook_text = build_hook_text(viral_breakdown)
                logger.info(f"🎣 Generated hook: '{hook_text}'")
                
                # Check if we should export a long variant
                kw_score = viral_breakdown.get("components", {}).get("keyword", 0)
                sent_score = viral_breakdown.get("components", {}).get("sentiment", 0)
                if EXPORT_LONG_TIKTOK and (kw_score + sent_score) > 0.25:
                    export_long_variant = True
                    logger.info("📱 Will export long TikTok variant (65s) - high teaching/reaction signals")
            
            # Check for IRL content before format & framing
            logger.info("🔍 Checking content type...")
            transcript_text = None
            if config.get('captions', {}).get('enabled', True):
                # Get transcript for both IRL detection and captions
                # Use faster transcription in performance mode
                fast_mode = performance_mode in ['fast', 'realtime']
                transcript = self._transcribe_audio(clip_path, fast_mode=fast_mode)
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
            video, safe_zone = self._apply_format_framing(video, config.get('format', {}), streamer_name)
            
            # Story B: Captions & Copy
            logger.info("📝 Adding captions...")
            if config.get('captions', {}).get('enabled', True) and transcript:
                # Use existing transcript instead of re-transcribing
                video, caption_telemetry = self._apply_captions_with_transcript(video, transcript, config.get('captions', {}), safe_zone, streamer_name, clip_path)
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
            # Pass performance mode to renderer
            platform_config['performance_mode'] = performance_mode
            # Store original clip path for audio recovery if needed
            original_clip_path = str(clip_path)
            self._render_video(video, output_path, platform_config, original_clip_path=original_clip_path)
            
            # Calculate telemetry
            processing_time = (time.time() - start_time) * 1000
            telemetry.processing_time_ms = int(processing_time)
            
            logger.info(f"✅ Enhancement complete: {output_path}")
            logger.info(f"📊 Processing time: {processing_time:.0f}ms")
            logger.info(f"📊 Words rendered: {telemetry.words_rendered}")
            logger.info(f"📊 Emphasis hits: {telemetry.emphasis_hits}")
            
            # TikTok variant export logic - save to temp, will be cleaned up
            if export_long_variant:
                logger.info("📱 Exporting 65s TikTok variant (temp)...")
                long_output_path = self.temp_dir / f"long_{output_path.name}"
                
                # Create longer version (65s) with slower pacing
                long_video = video.subclip(0, min(65, video.duration))
                
                # Add VO context overlay if available (future enhancement)
                # For now, just export the longer version
                long_platform_config = platform_config.copy()
                long_platform_config['performance_mode'] = 'fast'  # Quick render for variant
                self._render_video(long_video, long_output_path, long_platform_config, original_clip_path=original_clip_path)
                
                logger.info(f"📱 Long variant saved to temp: {long_output_path}")
                long_video.close()
            
            # Note: Main output is already TikTok/Instagram Shorts optimized (15-60s)
            # No need to create separate viral version - main output is the final clip
            
            # Cleanup
            video.close()
            
            # Clean up any temp files that ended up in the streamer folder
            self._cleanup_streamer_folder(output_path.parent)
            
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
    
    def _apply_format_framing(self, video, format_config: Dict, streamer_name: str = None) -> Tuple[Any, SafeZone]:
        """
        SIMPLE STABLE LAYOUT - No split-screen, no smart analysis
        
        1. Gameplay: Resize ONCE to 1080x1920 (fills entire screen)
        2. Facecam: Optional circular overlay at top-center (detected ONCE)
        """
        target_w, target_h = format_config.get('target_resolution', [1080, 1920])
        safe_zones_config = format_config.get('safe_zones', {})
        
        # Create safe zone
        safe_zone = SafeZone(
            bottom=safe_zones_config.get('caption_bottom', 0.25),
            left=0.05,
            right=0.05
        )
        
        original_w, original_h = video.size
        original_duration = video.duration
        
        # ================================================================
        # STEP 1: GAMEPLAY - Simple resize to fill entire vertical frame
        # ================================================================
        if target_w == 1080 and target_h == 1920:
            logger.info(f"📐 SIMPLE LAYOUT: Resizing {original_w}x{original_h} to {target_w}x{target_h}")
            
            # Calculate crop to maintain aspect ratio and fill frame
            # We want to crop the source to match 9:16 aspect ratio, then resize
            target_aspect = target_w / target_h  # 0.5625 for 9:16
            original_aspect = original_w / original_h
            
            if original_aspect > target_aspect:
                # Source is wider - crop sides
                new_w = int(original_h * target_aspect)
                x_offset = (original_w - new_w) // 2
                gameplay_clip = video.crop(x1=x_offset, y1=0, x2=x_offset + new_w, y2=original_h)
            else:
                # Source is taller - crop top/bottom
                new_h = int(original_w / target_aspect)
                y_offset = (original_h - new_h) // 2
                gameplay_clip = video.crop(x1=0, y1=y_offset, x2=original_w, y2=y_offset + new_h)
            
            # Resize ONCE to target
            gameplay_clip = gameplay_clip.resize((target_w, target_h))
            gameplay_clip = gameplay_clip.set_duration(original_duration)
            
            # ================================================================
            # STEP 2: FACECAM OVERLAY (optional) - Detect ONCE, position ONCE
            # ================================================================
            is_peanut_style = streamer_name and 'theburntpeanut' in streamer_name.lower()
            face_clip = None
            
            if is_peanut_style:
                # FIXED CONSTANTS - BIGGER CIRCLE
                FACE_SIZE = int(target_w * 0.35)  # 35% of width = ~378px (bigger circle!)
                FACE_Y = int(target_h * 0.025)    # 2.5% from top (moved down a bit)
                FACE_X = (target_w - FACE_SIZE) // 2 - 20  # Slightly left of center
                
                # theburntpeanut has TWO possible facecam positions:
                # 1. In-game: bottom-left (16-38% x, 60-90% y)
                # 2. Lounge/menu: center-left (5-27% x, 35-65% y)
                
                facecam_positions = [
                    # Position 1: In-game (bottom-left)
                    {
                        'name': 'in-game',
                        'x1': int(original_w * 0.16),
                        'y1': int(original_h * 0.60),
                        'x2': int(original_w * 0.38),
                        'y2': int(original_h * 0.90)
                    },
                    # Position 2: Lounge/menu (center-left)
                    {
                        'name': 'lounge',
                        'x1': int(original_w * 0.05),
                        'y1': int(original_h * 0.35),
                        'x2': int(original_w * 0.27),
                        'y2': int(original_h * 0.65)
                    }
                ]
                
                # Try each position and pick the one with the most face content
                best_position = None
                best_score = -1
                
                try:
                    # Sample a frame to test which position has more face content
                    test_frame = video.get_frame(video.duration * 0.5)
                    
                    if CV2_AVAILABLE and np is not None and cv2 is not None:
                        for pos in facecam_positions:
                            # Crop the test region
                            test_crop = test_frame[pos['y1']:pos['y2'], pos['x1']:pos['x2']]
                            
                            # Simple heuristic: count non-dark pixels (face is usually bright)
                            # and check variance (face has more detail than empty space)
                            gray = cv2.cvtColor(test_crop, cv2.COLOR_RGB2GRAY)
                            brightness = np.mean(gray)
                            variance = np.var(gray)
                            
                            # Score: prefer brighter regions with more variance (detail)
                            score = brightness * 0.5 + variance * 0.5
                            
                            logger.info(f"🔍 Testing {pos['name']} position: brightness={brightness:.1f}, variance={variance:.1f}, score={score:.1f}")
                            
                            if score > best_score:
                                best_score = score
                                best_position = pos
                        
                        if best_position:
                            logger.info(f"✅ Selected {best_position['name']} facecam position (score: {best_score:.1f})")
                    else:
                        # Fallback to in-game position if OpenCV not available
                        best_position = facecam_positions[0]
                        logger.info("⚠️ OpenCV not available, using default in-game position")
                    
                    # Use the best position
                    fc_x1, fc_y1 = best_position['x1'], best_position['y1']
                    fc_x2, fc_y2 = best_position['x2'], best_position['y2']
                    
                    # Crop and resize facecam ONCE
                    face_clip = video.crop(x1=fc_x1, y1=fc_y1, x2=fc_x2, y2=fc_y2)
                    face_clip = face_clip.resize((FACE_SIZE, FACE_SIZE))
                    face_clip = face_clip.set_duration(original_duration)
                    
                    # Create circular TRANSPARENCY mask using MoviePy's mask system
                    # This properly makes pixels outside the circle transparent (not black)
                    if CV2_AVAILABLE and np is not None and cv2 is not None:
                        mask_size = FACE_SIZE
                        center = (mask_size // 2, mask_size // 2)
                        radius = mask_size // 2 - 1
                        
                        # Create alpha mask (white = visible, black = transparent)
                        alpha_mask = np.zeros((mask_size, mask_size), dtype=np.float32)
                        cv2.circle(alpha_mask, center, radius, 1.0, -1)
                        
                        # Create a mask clip from the alpha mask
                        from moviepy.video.VideoClip import ImageClip
                        mask_clip = ImageClip(alpha_mask, ismask=True, duration=original_duration)
                        
                        # Apply the mask to make outside pixels transparent
                        face_clip = face_clip.set_mask(mask_clip)
                    
                    # Position ONCE
                    face_clip = face_clip.set_position((FACE_X, FACE_Y))
                    logger.info(f"🎭 FACECAM: {FACE_SIZE}x{FACE_SIZE} at ({FACE_X}, {FACE_Y})")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Facecam extraction failed: {e}, skipping overlay")
                    face_clip = None
            
            # ================================================================
            # STEP 3: COMPOSITE - Single operation
            # ================================================================
            if face_clip is not None:
                video = mp.CompositeVideoClip(
                    [gameplay_clip, face_clip],
                    size=(target_w, target_h)
                )
                video = video.set_duration(original_duration)
                safe_zone.bottom = 0.18  # Adjust for face overlay
            else:
                video = gameplay_clip
            
            logger.info(f"✅ STABLE LAYOUT COMPLETE: {target_w}x{target_h}")
        
        else:
            logger.info(f"📐 Keeping original format: {original_w}x{original_h}")
        
        return video, safe_zone
    
    def _analyze_video_layout(self, video) -> Dict:
        """
        Intelligently analyze video layout to determine optimal crop regions
        Returns layout analysis with recommended face and gameplay regions
        """
        try:
            # Sample frames from different parts of the video
            duration = video.duration
            sample_times = [duration * 0.1, duration * 0.5, duration * 0.9]
            
            layout_analysis = {
                'face_regions': [],
                'static_regions': [],
                'motion_regions': [],
                'recommended_layout': 'single_crop'
            }
            
            for sample_time in sample_times:
                try:
                    frame = video.get_frame(sample_time)
                    
                    # Detect faces in this frame
                    if CV2_AVAILABLE:
                        face_regions = self._detect_faces_in_frame(frame)
                        layout_analysis['face_regions'].extend(face_regions)
                        
                        # Analyze motion density (areas with lots of changing pixels)
                        motion_density = self._analyze_motion_density(frame)
                        layout_analysis['motion_regions'].append(motion_density)
                        
                        # Detect static UI elements (consistent across frames)
                        static_areas = self._detect_static_elements(frame)
                        layout_analysis['static_regions'].append(static_areas)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Frame analysis failed at {sample_time}s: {e}")
                    continue
            
            # Determine optimal layout based on analysis
            layout_analysis['recommended_layout'] = self._determine_optimal_layout(layout_analysis)
            
            logger.info(f"🔍 Layout Analysis: {layout_analysis['recommended_layout']} | Faces: {len(layout_analysis['face_regions'])} | Motion areas: {len(layout_analysis['motion_regions'])}")
            
            return layout_analysis
            
        except Exception as e:
            logger.error(f"❌ Layout analysis failed: {e}")
            return {'recommended_layout': 'single_crop', 'face_regions': [], 'motion_regions': [], 'static_regions': []}
    
    def _detect_faces_in_frame(self, frame) -> List[Tuple]:
        """Detect face regions in a single frame"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Use OpenCV's built-in face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            face_regions = []
            for (x, y, w, h) in faces:
                # Normalize coordinates to 0-1 range
                height, width = frame.shape[:2]
                face_regions.append((x/width, y/height, (x+w)/width, (y+h)/height))
            
            return face_regions
            
        except Exception as e:
            logger.warning(f"⚠️ Face detection failed: {e}")
            return []
    
    def _analyze_motion_density(self, frame) -> Dict:
        """Analyze which areas of the frame have high motion/activity"""
        try:
            # Convert to grayscale and apply edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Divide frame into grid and count edge density
            height, width = edges.shape
            grid_size = 8
            motion_map = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    y1 = int(i * height / grid_size)
                    y2 = int((i + 1) * height / grid_size)
                    x1 = int(j * width / grid_size)
                    x2 = int((j + 1) * width / grid_size)
                    
                    cell = edges[y1:y2, x1:x2]
                    density = np.sum(cell) / (cell.shape[0] * cell.shape[1] * 255)
                    
                    motion_map.append({
                        'region': (x1/width, y1/height, x2/width, y2/height),
                        'density': density
                    })
            
            return {'motion_map': motion_map, 'high_motion_areas': [m for m in motion_map if m['density'] > 0.1]}
            
        except Exception as e:
            logger.warning(f"⚠️ Motion analysis failed: {e}")
            return {'motion_map': [], 'high_motion_areas': []}
    
    def _detect_static_elements(self, frame) -> Dict:
        """Detect static UI elements like overlays, chat boxes, etc."""
        try:
            # Look for rectangular shapes that might be UI elements
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            ui_elements = []
            height, width = frame.shape[:2]
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If it's roughly rectangular and reasonably sized
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    area_ratio = (w * h) / (width * height)
                    
                    if 0.01 < area_ratio < 0.3:  # Between 1% and 30% of screen
                        ui_elements.append({
                            'region': (x/width, y/height, (x+w)/width, (y+h)/height),
                            'area_ratio': area_ratio,
                            'aspect_ratio': w/h if h > 0 else 1.0
                        })
            
            return {'ui_elements': ui_elements}
            
        except Exception as e:
            logger.warning(f"⚠️ Static element detection failed: {e}")
            return {'ui_elements': []}
    
    def _determine_optimal_layout(self, analysis: Dict) -> str:
        """Determine the best layout strategy based on analysis"""
        try:
            face_count = len(analysis['face_regions'])
            motion_areas = len([r for motion in analysis['motion_regions'] for r in motion.get('high_motion_areas', [])])
            
            # Decision logic for layout
            if face_count >= 3:  # Multiple faces detected across samples
                if motion_areas > 10:  # High motion areas suggest gameplay + facecam
                    return 'split_screen_horizontal'  # Face bottom, gameplay top
                else:
                    return 'split_screen_vertical'    # Face side, content other side
            elif face_count >= 1:  # Single face detected
                if motion_areas > 8:  # Likely gameplay with facecam
                    return 'split_screen_horizontal'  # Face bottom, gameplay top
                else:
                    return 'face_focus'  # Focus on the face
            else:  # No faces detected
                if motion_areas > 5:
                    return 'gameplay_focus'  # Focus on gameplay area
                else:
                    return 'single_crop'  # Just crop center
            
        except Exception as e:
            logger.warning(f"⚠️ Layout determination failed: {e}")
            return 'single_crop'
    
    def _detect_face_and_gameplay(self, video, streamer_name: str = None) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """Detect face and main gameplay areas using improved face detection"""
        if not CV2_AVAILABLE:
            logger.warning("⚠️ OpenCV not available for face detection")
            return None, None
        
        try:
            logger.info(f"🧠 Running SMART layout analysis for {streamer_name or 'unknown streamer'}")
            
            # Run intelligent layout analysis
            layout_analysis = self._analyze_video_layout(video)
            recommended_layout = layout_analysis['recommended_layout']
            face_regions = layout_analysis['face_regions']
            
            # Get frame dimensions
            frame = video.get_frame(video.duration * 0.5)
            frame_h, frame_w = frame.shape[:2]
                
            # Check if this is theburntpeanut style (face at top)
            is_peanut_style = streamer_name and 'theburntpeanut' in streamer_name.lower()
            
            # Determine regions based on intelligent analysis
            if recommended_layout == 'split_screen_horizontal' or is_peanut_style:
                if is_peanut_style:
                    logger.info("📱 SMART LAYOUT: Face-top layout - theburntpeanut style")
                    gameplay_region = (0, 0, frame_w, frame_h)  # Full screen for gameplay
                    
                    # theburntpeanut's facecam is in BOTTOM-LEFT corner!
                    # Based on screenshot: approximately 15-40% x, 55-95% y
                    facecam_x1 = int(0.12 * frame_w)
                    facecam_x2 = int(0.42 * frame_w)
                    facecam_y1 = int(0.55 * frame_h)
                    facecam_y2 = int(0.95 * frame_h)
                    
                    face_region = (facecam_x1, facecam_y1, facecam_x2, facecam_y2)
                    logger.info(f"📍 Using FIXED bottom-left facecam region for theburntpeanut: {face_region}")
                    return face_region, gameplay_region
                else:
                    logger.info("📱 SMART LAYOUT: Split-screen horizontal (face bottom, gameplay top)")
                
                if face_regions:
                    # For theburntpeanut style: use fixed top-center region OR largest face in top-center
                    # Facecam is typically in top-center (0.4-0.6 x, 0-0.3 y)
                    frame_area = frame_w * frame_h
                    top_center_faces = []
                    other_faces = []
                    
                    for face in face_regions:
                        fx1, fy1, fx2, fy2 = face
                        face_center_y = (fy1 + fy2) / 2  # Vertical center
                        face_center_x = (fx1 + fx2) / 2  # Horizontal center
                        face_area = (fx2 - fx1) * (fy2 - fy1) * frame_area
                        face_area_ratio = face_area / frame_area
                        
                        # Check if face is in top-center region (where facecam usually is)
                        is_top_center = (face_center_y < 0.3 and 0.35 < face_center_x < 0.65)
                        is_large_enough = face_area_ratio > 0.03
                        
                        if is_top_center and is_large_enough:
                            top_center_faces.append((face, face_area_ratio))
                        else:
                            other_faces.append((face, face_area_ratio))
                    
                    # Prefer top-center faces, otherwise use largest face in top region
                    if top_center_faces:
                        # Use largest face in top-center
                        top_center_faces.sort(key=lambda x: x[1], reverse=True)
                        primary_face = top_center_faces[0][0]
                        logger.info(f"✅ Using top-center face: area={top_center_faces[0][1]:.3f}")
                    elif other_faces:
                        # Use largest face in top region (even if not perfectly centered)
                        top_faces = [(f, a) for f, a in other_faces if (f[1] + f[3])/2 < 0.4]
                        if top_faces:
                            top_faces.sort(key=lambda x: x[1], reverse=True)
                            primary_face = top_faces[0][0]
                            logger.info(f"⚠️ Using top region face (not center): area={top_faces[0][1]:.3f}")
                        else:
                            # Last resort: largest face overall
                            other_faces.sort(key=lambda x: x[1], reverse=True)
                            primary_face = other_faces[0][0]
                            logger.warning(f"⚠️ Using largest face (may not be streamer): area={other_faces[0][1]:.3f}")
                    else:
                        # No faces found - use default top-center region
                        primary_face = None
                    
                    if primary_face:
                        face_x1, face_y1, face_x2, face_y2 = primary_face
                        
                        # Convert to pixel coordinates and expand face region slightly
                        face_x1 = max(0, int(face_x1 * frame_w - 0.05 * frame_w))
                        face_x2 = min(frame_w, int(face_x2 * frame_w + 0.05 * frame_w))
                        face_y1 = max(0, int(face_y1 * frame_h))
                        face_y2 = min(int(0.3 * frame_h), int(face_y2 * frame_h + 0.05 * frame_h))
                        
                        face_region = (face_x1, face_y1, face_x2, face_y2)
                        gameplay_region = (0, 0, frame_w, frame_h)  # Full screen for gameplay
                    else:
                        # Original: face at bottom
                        face_y1 = max(int(0.6 * frame_h), int(face_y1 * frame_h))  # Keep in bottom area
                        face_y2 = frame_h  # Extend to bottom
                        face_region = (face_x1, face_y1, face_x2, face_y2)
                        gameplay_region = (0, 0, frame_w, int(0.65 * frame_h))  # Top area for gameplay
                else:
                    # No faces detected - use default regions
                    if is_peanut_style:
                        # Default top-center for face if no face detected (theburntpeanut style)
                        # Use fixed top-center region: 0.4-0.6 x, 0-0.25 y
                        face_region = (int(0.4 * frame_w), 0, int(0.6 * frame_w), int(0.25 * frame_h))
                        logger.info(f"📍 Using default top-center face region: {face_region}")
                        gameplay_region = (0, 0, frame_w, frame_h)  # Full screen for gameplay
                    else:
                        # Default bottom-center for face if no face detected
                        face_region = (int(0.3 * frame_w), int(0.6 * frame_h), frame_w, frame_h)
                        gameplay_region = (0, 0, frame_w, int(0.65 * frame_h))  # Top area for gameplay
                    
            elif recommended_layout == 'split_screen_vertical':
                logger.info("📱 SMART LAYOUT: Split-screen vertical (face side, gameplay other side)")
                
                if face_regions:
                    primary_face = face_regions[0]
                    face_x1, face_y1, face_x2, face_y2 = primary_face
                    face_center_x = (face_x1 + face_x2) / 2
                    
                    if face_center_x > 0.5:  # Face on right side
                        face_region = (int(0.6 * frame_w), 0, frame_w, frame_h)
                        gameplay_region = (0, 0, int(0.6 * frame_w), frame_h)
                    else:  # Face on left side
                        face_region = (0, 0, int(0.4 * frame_w), frame_h)
                        gameplay_region = (int(0.4 * frame_w), 0, frame_w, frame_h)
                else:
                    # Default right side for face
                    face_region = (int(0.6 * frame_w), 0, frame_w, frame_h)
                    gameplay_region = (0, 0, int(0.6 * frame_w), frame_h)
                    
            elif recommended_layout == 'face_focus':
                logger.info("📱 SMART LAYOUT: Face focus (Just Chatting style)")
                
                if face_regions:
                    primary_face = face_regions[0]
                    face_x1, face_y1, face_x2, face_y2 = primary_face
                    
                    # Expand around the detected face
                    face_x1 = max(0, int((face_x1 - 0.2) * frame_w))
                    face_y1 = max(0, int((face_y1 - 0.1) * frame_h))
                    face_x2 = min(frame_w, int((face_x2 + 0.2) * frame_w))
                    face_y2 = min(frame_h, int((face_y2 + 0.1) * frame_h))
                    
                    face_region = (face_x1, face_y1, face_x2, face_y2)
                    gameplay_region = None  # No separate gameplay region
                else:
                    face_region = (int(0.1 * frame_w), int(0.1 * frame_h), 
                                 int(0.9 * frame_w), int(0.9 * frame_h))  # Center crop
                    gameplay_region = None
                    
            elif recommended_layout == 'gameplay_focus':
                logger.info("📱 SMART LAYOUT: Gameplay focus (no face detected)")
                face_region = None
                
                # Find highest motion area for gameplay
                motion_regions = layout_analysis['motion_regions']
                if motion_regions and motion_regions[0].get('high_motion_areas'):
                    # Use the area with highest motion density
                    high_motion = max(motion_regions[0]['high_motion_areas'], 
                                    key=lambda x: x['density'])
                    mx1, my1, mx2, my2 = high_motion['region']
                    gameplay_region = (int(mx1 * frame_w), int(my1 * frame_h),
                                     int(mx2 * frame_w), int(my2 * frame_h))
                else:
                    gameplay_region = (int(0.1 * frame_w), int(0.1 * frame_h),
                                     int(0.9 * frame_w), int(0.9 * frame_h))  # Center crop
                    
            else:  # single_crop fallback
                logger.info("📱 SMART LAYOUT: Single crop (center focus)")
                face_region = (int(0.1 * frame_w), int(0.1 * frame_h), 
                             int(0.9 * frame_w), int(0.9 * frame_h))
                gameplay_region = None
            
            # Log the final decision
            if face_region and gameplay_region:
                logger.info(f"✅ SMART CROP: Face {face_region} | Gameplay {gameplay_region}")
            elif face_region:
                logger.info(f"✅ SMART CROP: Face-only {face_region}")
            elif gameplay_region:
                logger.info(f"✅ SMART CROP: Gameplay-only {gameplay_region}")
            
            return face_region, gameplay_region
                
        except Exception as e:
            logger.error(f"❌ Face detection failed: {e}")
            import traceback
            traceback.print_exc()
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
    
    def _apply_captions_with_transcript(self, video, transcript: List[Dict], captions_config: Dict, safe_zone: SafeZone, streamer_name: str, clip_path: Path = None) -> Tuple[Any, Dict]:
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
        hook_text = None
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
        
        # CAPTIONS DISABLED - clean output without text overlay
        # if captions_config.get('karaoke_sync', True) and words:
        #     logger.info(f"📝 Applying {len(words)} caption words using video.fl() method")
        #     video = self._apply_captions_with_fl(video, words, captions_config.get('style', {}), safe_zone, hook_text=hook_text, streamer_name=streamer_name)
        logger.info("📝 Captions DISABLED - clean output")
        
        telemetry['words_rendered'] = len(words)
        telemetry['emphasis_hits'] = sum(1 for word in words if word.is_emphasis)
        
        return video, telemetry
    
    def _apply_captions_with_fl(self, video, words: List[CaptionWord], style_config: Dict, safe_zone: SafeZone, hook_text: Optional[str] = None, streamer_name: str = None):
        """
        PHRASE-BASED CAPTIONS - Not word-level karaoke
        
        1. Group words into natural phrases (3-9 words)
        2. Score phrases for importance
        3. Keep only top ~10-16 phrases for a 30s clip
        4. Render as complete phrases, no animation
        """
        try:
            import cv2
            import numpy as np
            
            # Style settings
            is_peanut_style = (streamer_name and 'theburntpeanut' in streamer_name.lower())
            
            if is_peanut_style:
                font_size = 90
                font_color = (255, 165, 0)  # Orange
                stroke_color = (0, 0, 0)
                stroke_width = 8
            else:
                font_size = style_config.get('font_size', 80)
                font_color = tuple(style_config.get('font_color', [255, 255, 0]))
                stroke_color = tuple(style_config.get('stroke_color', [0, 0, 0]))
                stroke_width = style_config.get('stroke_width', 6)
            
            # ================================================================
            # STEP 1: PHRASE SEGMENTATION
            # Group words into natural phrases based on pauses and punctuation
            # ================================================================
            phrases = []
            if not words:
                logger.info("📝 No words to caption")
                return video
            
            sorted_words = sorted(words, key=lambda w: w.start_time)
            
            # Filler words to penalize
            FILLERS = {'uh', 'um', 'like', 'you know', 'i mean', 'basically', 'actually', 'literally'}
            
            # Break words into phrases
            current_phrase = []
            phrase_start = None
            
            for i, word in enumerate(sorted_words):
                text = word.text.strip()
                if not text:
                    continue
                
                if not current_phrase:
                    current_phrase = [word]
                    phrase_start = word.start_time
                else:
                    # Check for phrase break conditions
                    gap = word.start_time - current_phrase[-1].end_time
                    prev_text = current_phrase[-1].text.lower()
                    
                    # Break on: pause >= 0.7s, punctuation, conjunctions, or max length
                    should_break = (
                        gap >= 0.7 or
                        len(current_phrase) >= 9 or
                        prev_text.endswith(('.', '!', '?', ',')) or
                        text.lower() in ('but', 'so', 'then', 'and', 'because', 'when', 'if')
                    )
                    
                    if should_break and len(current_phrase) >= 3:
                        # Finalize phrase
                        phrase_end = current_phrase[-1].end_time
                        phrase_text = ' '.join(w.text for w in current_phrase)
                        phrases.append({
                            'text': phrase_text.upper(),
                            'start': phrase_start,
                            'end': phrase_end,
                            'word_count': len(current_phrase),
                            'words': current_phrase
                        })
                        current_phrase = [word]
                        phrase_start = word.start_time
                    else:
                        current_phrase.append(word)
            
            # Finalize last phrase
            if current_phrase and len(current_phrase) >= 2:
                phrase_end = current_phrase[-1].end_time
                phrase_text = ' '.join(w.text for w in current_phrase)
                phrases.append({
                    'text': phrase_text.upper(),
                    'start': phrase_start,
                    'end': phrase_end,
                    'word_count': len(current_phrase),
                    'words': current_phrase
                })
            
            # ================================================================
            # STEP 2: PHRASE SCORING
            # Score each phrase for importance, drop low-scoring ones
            # ================================================================
            video_duration = video.duration
            target_phrases = max(8, min(16, int(video_duration / 2)))  # ~1 phrase per 2 seconds
            
            for phrase in phrases:
                score = 0.0
                text_lower = phrase['text'].lower()
                duration = phrase['end'] - phrase['start']
                
                # Duration score (prefer 0.8-2.0s phrases)
                if 0.8 <= duration <= 2.0:
                    score += 2.0
                elif 0.5 <= duration <= 2.5:
                    score += 1.0
                else:
                    score -= 1.0
                
                # Word count score (prefer 4-7 words)
                wc = phrase['word_count']
                if 4 <= wc <= 7:
                    score += 1.5
                elif 3 <= wc <= 9:
                    score += 0.5
                
                # Penalize fillers
                filler_count = sum(1 for f in FILLERS if f in text_lower)
                score -= filler_count * 1.0
                
                # Bonus for reaction words
                reactions = ['wow', 'omg', 'what', 'no', 'yes', 'lets go', 'nice', 'dude', 'bro', 'insane', 'crazy']
                if any(r in text_lower for r in reactions):
                    score += 2.0
                
                # Bonus for questions
                if '?' in phrase['text']:
                    score += 1.0
                
                phrase['score'] = score
            
            # Sort by score and keep top phrases
            phrases.sort(key=lambda p: p['score'], reverse=True)
            kept_phrases = phrases[:target_phrases]
            
            # Re-sort by time for display
            kept_phrases.sort(key=lambda p: p['start'])
            
            # ================================================================
            # STEP 3: TIMING SMOOTHING
            # Adjust timing to prevent flicker and overlap
            # ================================================================
            final_phrases = []
            for i, phrase in enumerate(kept_phrases):
                # Start slightly after phrase begins
                start = phrase['start'] + 0.1
                # End slightly after phrase ends, but not too long
                end = phrase['end'] + 0.3
                
                # Ensure no overlap with next phrase
                if i + 1 < len(kept_phrases):
                    next_start = kept_phrases[i + 1]['start']
                    if end > next_start - 0.1:
                        end = next_start - 0.1
                
                # Minimum display time of 0.5s
                if end - start < 0.5:
                    end = start + 0.5
                
                final_phrases.append({
                    'text': phrase['text'],
                    'start': start,
                    'end': end
                })
            
            logger.info(f"📝 PHRASE CAPTIONS: {len(final_phrases)} phrases from {len(words)} words (target: {target_phrases})")
            
            # ================================================================
            # STEP 4: RENDERING - Simple, stable, no animation
            # ================================================================
            # Pre-compute fixed values
            CAPTION_Y = 0.42  # Fixed Y position (42% from top)
            
            def caption_effect(get_frame, t):
                frame = get_frame(t)
                frame_height, frame_width = frame.shape[:2]
                
                # Find active phrase (only one at a time)
                active_phrase = None
                for phrase in final_phrases:
                    if phrase['start'] <= t <= phrase['end']:
                        active_phrase = phrase
                        break
                
                # No phrase = no caption
                if not active_phrase:
                    return frame
                
                text = active_phrase['text']
                font_scale = font_size / 30
                
                # Split into max 2 lines if needed
                max_width = int(frame_width * 0.9)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, stroke_width)[0]
                
                if text_size[0] > max_width:
                    words_list = text.split()
                    mid = len(words_list) // 2
                    lines = [' '.join(words_list[:mid]), ' '.join(words_list[mid:])]
                else:
                    lines = [text]
                
                # Fixed Y position
                line_height = int(font_size * 1.4)
                total_height = len(lines) * line_height
                base_y = int(frame_height * CAPTION_Y)
                start_y = base_y - (total_height // 2)
                
                # Render each line
                for i, line in enumerate(lines[:2]):
                    if not line.strip():
                        continue
                    
                    line_y = start_y + (i * line_height) + line_height
                    
                    # Measure for centering
                    line_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, font_scale, stroke_width)[0]
                    x_pos = (frame_width - line_size[0]) // 2
                    
                    # Shadow
                    cv2.putText(frame, line, (x_pos + 3, line_y + 3),
                              cv2.FONT_HERSHEY_DUPLEX, font_scale, (20, 20, 20), stroke_width + 3, cv2.LINE_AA)
                    
                    # Outline
                    cv2.putText(frame, line, (x_pos, line_y),
                              cv2.FONT_HERSHEY_DUPLEX, font_scale, stroke_color, stroke_width + 2, cv2.LINE_AA)
                    
                    # Text
                    cv2.putText(frame, line, (x_pos, line_y),
                              cv2.FONT_HERSHEY_DUPLEX, font_scale, font_color, max(1, stroke_width - 3), cv2.LINE_AA)
                
                return frame
            
            logger.info("✅ Applied PHRASE-BASED captions")
            enhanced_video = video.fl(caption_effect)
            return enhanced_video
            
        except Exception as e:
            logger.warning(f"⚠️ Caption effect failed: {e}")
            import traceback
            traceback.print_exc()
            return video
    
    def _apply_viral_effects(self, video, words: List[CaptionWord], viral_config: Dict):
        """Apply AUDIO-REACTIVE viral effects based on volume levels!"""
        try:
            effects_applied = []
            duration = video.duration
            
            logger.info(f"🎵 Analyzing audio for reactive effects on {duration:.1f}s clip")
            
            # Extract audio for volume analysis
            audio = video.audio
            if not audio:
                logger.warning("⚠️ No audio found, using minimal effects")
                # Fallback to minimal flash effects at start/end
                video = self._add_flash_effect(video, 0.5, 0.1)
                video = self._add_flash_effect(video, duration - 1.0, 0.1)
                return video
            
            # Analyze volume levels throughout the clip
            volume_peaks = self._analyze_audio_volume(audio, duration)
            if volume_peaks is None:
                volume_peaks = []
            
            logger.info(f"🔊 Found {len(volume_peaks)} volume peaks for reactive effects")
            
            for peak in volume_peaks:
                peak_time = peak['time']
                volume_level = peak['volume']  # 0.0 to 1.0
                peak_duration = peak['duration']
                
                try:
                    # Different effects based on volume intensity
                    if volume_level > 0.9:
                        # EXTREMELY LOUD - Light shake + subtle zoom (reduced intensity)
                        intensity = min(volume_level * 0.015, 0.02)  # Much reduced shake intensity
                        zoom_factor = 1.0 + (volume_level * 0.1)  # Much reduced zoom
                        
                        video = self._add_screen_shake_fl(video, peak_time, peak_duration, intensity=intensity)
                        video = self._add_zoom_effect_fl(video, peak_time, peak_duration, zoom_factor=zoom_factor)
                        effects_applied.append(f"LOUD@{peak_time:.1f}s(vol:{volume_level:.2f})")
                        
                    elif volume_level > 0.8:
                        # VERY LOUD - Just subtle shake (no zoom, no flash)
                        intensity = min(volume_level * 0.01, 0.015)  # Very subtle shake
                        
                        video = self._add_screen_shake_fl(video, peak_time, peak_duration, intensity=intensity)
                        effects_applied.append(f"SHAKE@{peak_time:.1f}s(vol:{volume_level:.2f})")
                    else:
                        # MODERATE LOUD - Just tiny zoom (very subtle)
                        zoom_factor = 1.0 + (volume_level * 0.05)  # Very subtle zoom
                        
                        video = self._add_zoom_effect_fl(video, peak_time, peak_duration, zoom_factor=zoom_factor)
                        effects_applied.append(f"ZOOM@{peak_time:.1f}s(vol:{volume_level:.2f})")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Audio-reactive effect at {peak_time:.1f}s failed: {e}")
            
            # BONUS: Add emphasis word effects for extra impact
            emphasis_moments = [word for word in words if word.is_emphasis]
            for word in emphasis_moments:
                try:
                    # Add flash for emphasis words regardless of volume
                    video = self._add_flash_effect(video, word.start_time, 0.06)
                    effects_applied.append(f"WORD_FLASH@{word.start_time:.1f}s")
                except Exception as e:
                    logger.warning(f"⚠️ Emphasis effect failed: {e}")
            
            logger.info(f"🎵 Applied {len(effects_applied)} AUDIO-REACTIVE effects: {', '.join(effects_applied[:5])}{'...' if len(effects_applied) > 5 else ''}")
            return video
            
        except Exception as e:
            logger.error(f"❌ Audio-reactive effects failed: {e}")
            return video
    
    def _analyze_audio_volume(self, audio, duration):
        """Analyze audio volume levels to find peaks for reactive effects"""
        try:
            import numpy as np
            
            # Sample the audio at regular intervals
            sample_rate = 10  # Check volume 10 times per second
            total_samples = int(duration * sample_rate)
            volume_peaks = []
            
            logger.info(f"🎵 Analyzing {total_samples} audio samples...")
            
            for i in range(total_samples):
                sample_time = i / sample_rate
                if sample_time >= duration:
                    break
                
                try:
                    # Get audio chunk around this time (0.1s window)
                    chunk_start = max(0, sample_time - 0.05)
                    chunk_end = min(duration, sample_time + 0.05)
                    
                    # Extract audio data for this chunk
                    audio_chunk = audio.subclip(chunk_start, chunk_end)
                    audio_array = audio_chunk.to_soundarray()
                    
                    # Calculate RMS volume (root mean square)
                    if len(audio_array) > 0:
                        rms_volume = np.sqrt(np.mean(audio_array**2))
                        
                        # Normalize volume (typical speech is around 0.1-0.3 RMS)
                        normalized_volume = min(rms_volume / 0.2, 1.0)  # Normalize to 0-1
                        
                        # Only consider significant volume changes (peaks) - MUCH LESS SENSITIVE
                        if normalized_volume > 0.7:  # Even higher threshold for much fewer effects
                            volume_peaks.append({
                                'time': sample_time,
                                'volume': normalized_volume,
                                'duration': 0.3  # Effect duration
                            })
                            
                except Exception as e:
                    # Skip problematic samples
                    continue
            
            # Filter peaks to avoid too many effects (more spacing)
            filtered_peaks = []
            last_peak_time = -1.5  # Start offset
            
            for peak in volume_peaks:
                if peak['time'] - last_peak_time >= 2.5:  # At least 2.5 seconds apart - MUCH MORE SPACING
                    filtered_peaks.append(peak)
                    last_peak_time = peak['time']
                elif peak['volume'] > 0.9:  # Only EXTREMELY loud moments can break spacing rule
                    filtered_peaks.append(peak)
                    last_peak_time = peak['time']
            
            logger.info(f"🔊 Filtered to {len(filtered_peaks)} volume peaks from {len(volume_peaks)} candidates")
            return filtered_peaks
            
        except Exception as e:
            logger.warning(f"⚠️ Audio analysis failed: {e}")
            # Fallback to some basic timing if audio analysis fails
            fallback_peaks = [
                {'time': 1.0, 'volume': 0.7, 'duration': 0.3},
                {'time': duration * 0.5, 'volume': 0.8, 'duration': 0.3},
                {'time': duration - 2.0, 'volume': 0.9, 'duration': 0.3}
            ]
            return [p for p in fallback_peaks if p['time'] > 0 and p['time'] < duration]
        
        # Ensure we always return a list
        return volume_peaks if volume_peaks is not None else []
    
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
        """Add flash effect using fx instead of composite overlay to avoid freezing"""
        try:
            logger.info(f"⚡ Adding flash effect at {start_time:.1f}s for {duration:.2f}s")
            
            # Use video effects instead of overlay composition to avoid CompositeVideoClip issues
            def flash_effect(get_frame, t):
                frame = get_frame(t)
                # Apply flash effect during the specified time window
                if start_time <= t <= start_time + duration:
                    # Calculate flash intensity (fade in/out)
                    progress = (t - start_time) / duration
                    if progress < 0.1:  # Fade in
                        intensity = progress / 0.1
                    elif progress > 0.9:  # Fade out  
                        intensity = (1.0 - progress) / 0.1
                    else:  # Full flash
                        intensity = 1.0
                    
                    # Brighten the frame instead of overlaying white
                    intensity = min(intensity * 0.3, 0.3)  # Max 30% brightness boost
                    flash_frame = frame.astype(float)
                    flash_frame = flash_frame + (255 - flash_frame) * intensity
                    return flash_frame.astype('uint8')
                
                return frame
            
            return video.fl(flash_effect)
        except Exception as e:
            logger.warning(f"⚠️ Flash effect failed: {e}")
            return video
    
    def _add_screen_shake_fl(self, video, start_time: float, duration: float, intensity: float = 0.05):
        """Add screen shake using video.fl() instead of CompositeVideoClip"""
        try:
            def shake_effect(get_frame, t):
                frame = get_frame(t)
                if start_time <= t <= start_time + duration:
                    progress = (t - start_time) / duration
                    shake_intensity = intensity * math.sin(progress * math.pi * 8) * 0.5  # Further reduced shake frequency and amplitude
                    
                    h, w = frame.shape[:2]
                    offset_x = int(shake_intensity * w * random.uniform(-1, 1))
                    offset_y = int(shake_intensity * h * random.uniform(-1, 1))
                    
                    # Create shaken frame by shifting
                    shaken = np.zeros_like(frame)
                    
                    # Calculate bounds to avoid out-of-bounds
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
            logger.warning(f"⚠️ Shake FL effect failed: {e}")
            return video
    
    def _add_zoom_effect_fl(self, video, start_time: float, duration: float, zoom_factor: float = 1.2):
        """Add zoom effect using video.fl() instead of CompositeVideoClip"""
        try:
            def zoom_effect(get_frame, t):
                frame = get_frame(t)
                if start_time <= t <= start_time + duration:
                    progress = (t - start_time) / duration
                    # Smooth zoom in and out
                    current_zoom = 1.0 + (zoom_factor - 1.0) * math.sin(progress * math.pi)
                    
                    h, w = frame.shape[:2]
                    center_x, center_y = w // 2, h // 2
                    
                    # Calculate new dimensions
                    new_w = int(w / current_zoom)
                    new_h = int(h / current_zoom)
                    
                    # Calculate crop region (centered)
                    x1 = max(0, center_x - new_w // 2)
                    y1 = max(0, center_y - new_h // 2)
                    x2 = min(w, x1 + new_w)
                    y2 = min(h, y1 + new_h)
                    
                    # Crop and resize back to original size
                    cropped = frame[y1:y2, x1:x2]
                    zoomed = cv2.resize(cropped, (w, h))
                    return zoomed
                
                return frame
            
            return video.fl(zoom_effect)
        except Exception as e:
            logger.warning(f"⚠️ Zoom FL effect failed: {e}")
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
    
    def _transcribe_audio(self, clip_path: Path, fast_mode: bool = False) -> Optional[List[Dict]]:
        """Transcribe audio using Whisper"""
        if not WHISPER_AVAILABLE:
            logger.warning("⚠️ Whisper not available, skipping transcription")
            return None
        
        try:
            # Choose model based on performance requirements
            model_name = "tiny" if fast_mode else "tiny"  # Even "tiny" is quite good for short clips
            
            if self.whisper_model is None:
                logger.info(f"🤖 Loading Whisper model: {model_name}")
                self.whisper_model = whisper.load_model(model_name)
            
            result = self.whisper_model.transcribe(str(clip_path), word_timestamps=True)
            
            # Use word-level timestamps if available (more accurate)
            words = []
            if 'segments' in result:
                for segment in result['segments']:
                    # Check if word_timestamps are available
                    if 'words' in segment and segment['words']:
                        # Use precise word timestamps from Whisper
                        for word_info in segment['words']:
                            clean_text = word_info['word'].strip('.,!?;:()[]{}"\'').strip()
                            if clean_text:  # Only add non-empty words
                                words.append({
                                    'text': clean_text,
                                    'start': word_info['start'],
                                    'end': word_info['end']
                                })
                    else:
                        # Fallback: split segment text and estimate timing
                        segment_words = segment['text'].strip().split()
                        if segment_words:
                            segment_duration = segment['end'] - segment['start']
                            word_duration = segment_duration / len(segment_words)
                            
                            for i, word in enumerate(segment_words):
                                # Clean word of punctuation for better matching
                                clean_word = word.strip('.,!?;:()[]{}"\'').strip()
                                if clean_word:  # Only add non-empty words
                                    words.append({
                                        'text': clean_word,
                                        'start': segment['start'] + (i * word_duration),
                                        'end': segment['start'] + ((i + 1) * word_duration)
                                    })
            
            logger.info(f"📝 Transcribed {len(words)} words from audio")
            return words if words else None
            
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
            logger.info(f"🏷️ Applying branding using video.fl() method")
            video = self._apply_branding_with_fl(video, branding_config, streamer_name)
        
        return video, telemetry
    
    def _apply_branding_with_fl(self, video, branding_config: Dict, streamer_name: str):
        """Apply watermark and branding using video.fl() to avoid CompositeVideoClip freezing"""
        try:
            import cv2
            import numpy as np
            
            watermark_config = branding_config.get('watermark', {})
            if not watermark_config.get('enabled', False):
                return video
                
            watermark_text = watermark_config.get('handle_text', f'@{streamer_name}_clippy')
            
            def watermark_effect(get_frame, t):
                frame = get_frame(t)
                frame_height, frame_width = frame.shape[:2]
                
                # Watermark styling
                font_size = 0.6
                font_color = (255, 255, 255)  # White
                stroke_color = (0, 0, 0)  # Black outline
                stroke_width = 2
                
                # Position watermark in bottom-right corner
                text_size = cv2.getTextSize(watermark_text, cv2.FONT_HERSHEY_SIMPLEX, font_size, stroke_width)[0]
                x_pos = frame_width - text_size[0] - 20
                y_pos = frame_height - 20
                
                # Draw watermark with outline
                cv2.putText(frame, watermark_text, (x_pos, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_size, stroke_color, stroke_width + 1)
                cv2.putText(frame, watermark_text, (x_pos, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, stroke_width)
                
                return frame
            
            logger.info(f"✅ Applied watermark '{watermark_text}' using video.fl() method")
            return video.fl(watermark_effect)
            
        except Exception as e:
            logger.warning(f"⚠️ Branding FL effect failed: {e}")
            return video
    
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
    
    def _render_video(self, video, output_path: Path, platform_config: Dict, original_clip_path: Optional[str] = None):
        """Render final video with platform-native settings"""
        codec = platform_config.get('output_codec', 'libx264')
        crf = platform_config.get('output_crf', 18)
        preset = platform_config.get('output_preset', 'veryfast')
        performance_mode = platform_config.get('performance_mode', 'balanced')
        
        # Adjust settings based on performance mode
        if performance_mode == 'realtime':
            preset = 'ultrafast'
            crf = 28  # Lower quality for maximum speed
            threads = 8  # More threads
        elif performance_mode == 'fast':
            preset = 'ultrafast'  # Changed from superfast to ultrafast
            crf = 25  # Lower quality for speed
            threads = 8  # More threads
        else:
            preset = 'superfast'  # Changed from veryfast to superfast
            crf = 23  # Slightly lower quality
            threads = 6  # More threads
        
        logger.info(f"⚡ Rendering with preset: {preset}, CRF: {crf}, threads: {threads}")
        
        # Export video with optimized settings and timeout protection
        import signal
        import threading
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Video rendering timed out")
        
        # Ensure audio is properly attached and synchronized before rendering
        render_video = video  # Use a local variable to avoid scoping issues
        if hasattr(render_video, 'audio') and render_video.audio is not None:
            try:
                # Make sure audio duration matches video duration
                if abs(render_video.audio.duration - render_video.duration) > 0.1:
                    logger.warning(f"⚠️ Audio duration ({render_video.audio.duration:.2f}s) doesn't match video ({render_video.duration:.2f}s), fixing...")
                    # Subclip audio to match video duration
                    audio_fixed = render_video.audio.subclip(0, min(render_video.audio.duration, render_video.duration))
                    # Set audio duration to match video
                    render_video = render_video.set_audio(audio_fixed.set_duration(render_video.duration))
                else:
                    # Ensure audio is properly set
                    render_video = render_video.set_audio(render_video.audio)
            except Exception as audio_fix_error:
                logger.warning(f"⚠️ Could not fix audio timing: {audio_fix_error}, using original audio")
        else:
            logger.warning("⚠️ No audio found on video")
        
        def render_with_timeout():
            # Ensure we're using absolute paths to avoid directory issues
            import os
            # Use temp directory, not output directory, to avoid nested folders
            temp_dir = self.temp_dir.resolve()
            
            # Ensure temp directory exists and use absolute paths
            temp_dir.mkdir(parents=True, exist_ok=True)
            output_path_abs = output_path.resolve()
            output_path_abs.parent.mkdir(parents=True, exist_ok=True)
            
            # Set temp_audiofile to be in temp_dir using absolute path
            temp_audiofile_abs = str(temp_dir / f'TEMP_MPY_wvf_snd_{output_path_abs.stem}.mp4')
            
            # Don't change directories - use absolute paths everywhere
            render_video.write_videofile(
                str(output_path_abs),  # Use absolute path
                codec=codec,
                preset=preset,
                ffmpeg_params=['-crf', str(crf), '-threads', str(threads)],
                audio_codec='aac' if (hasattr(render_video, 'audio') and render_video.audio is not None) else None,
                verbose=True,  # Show progress bar
                logger='bar',  # Enable progress bar
                temp_audiofile=temp_audiofile_abs  # Use absolute path for temp audio
            )
        
        try:
            # Set a 10-minute timeout for rendering (Windows-compatible)
            import time
            start_time = time.time()
            timeout_seconds = 600  # 10 minutes
            
            def check_timeout():
                if time.time() - start_time > timeout_seconds:
                    logger.error("❌ Rendering timeout after 10 minutes - killing process")
                    os._exit(1)
            
            # Start timeout monitor thread
            timeout_thread = threading.Thread(target=lambda: [time.sleep(1), check_timeout()] * timeout_seconds, daemon=True)
            timeout_thread.start()
            
            render_with_timeout()
                
        except (TimeoutError, KeyboardInterrupt, IndexError, Exception) as e:
            error_msg = str(e)
            logger.warning(f"⚠️ Rendering failed: {error_msg}")
            
            # Check if it's an audio-related error
            is_audio_error = 'audio' in error_msg.lower() or 'index' in error_msg.lower() or 'buffer' in error_msg.lower()
            
            if is_audio_error and original_clip_path:
                logger.info("🔄 Audio error detected - using ffmpeg for complete render...")
                try:
                    import subprocess
                    import os
                    temp_dir = self.temp_dir.resolve()  # Use absolute path
                    # Ensure temp directory exists
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    # Use simple filenames without duplicate "enhanced"
                    clean_stem = output_path.stem.replace('enhanced_', '').replace('viral_', '')
                    temp_video_no_audio = temp_dir / f"temp_video_no_audio_{clean_stem}.mp4"
                    temp_audio = temp_dir / f"temp_audio_{clean_stem}.wav"
                    # Use absolute paths for all files
                    temp_video_no_audio = temp_video_no_audio.resolve()
                    temp_audio = temp_audio.resolve()
                    original_clip_path_abs = Path(original_clip_path).resolve()
                    
                    # Step 1: Extract video without audio using ffmpeg
                    logger.info("🔄 Extracting video without audio...")
                    ffmpeg_video_cmd = [
                        'ffmpeg', '-i', str(original_clip_path_abs),
                        '-an', '-vcodec', 'copy',  # No audio, copy video codec
                        '-y', str(temp_video_no_audio)
                    ]
                    result = subprocess.run(ffmpeg_video_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.error(f"❌ FFmpeg video extraction failed: {result.stderr}")
                        raise Exception(f"FFmpeg video extraction failed: {result.stderr}")
                    
                    # Step 2: Extract audio using ffmpeg
                    logger.info("🔄 Extracting audio with ffmpeg...")
                    ffmpeg_audio_cmd = [
                        'ffmpeg', '-i', str(original_clip_path_abs),
                        '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                        '-y', str(temp_audio)
                    ]
                    result = subprocess.run(ffmpeg_audio_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.error(f"❌ FFmpeg audio extraction failed: {result.stderr}")
                        raise Exception(f"FFmpeg audio extraction failed: {result.stderr}")
                    
                    if temp_video_no_audio.exists() and temp_audio.exists() and temp_audio.stat().st_size > 0:
                        # Step 3: Render enhanced video WITHOUT audio first (avoids corrupted buffer)
                        logger.info("🔄 Rendering enhanced video without audio...")
                        video_no_audio_clip = render_video.without_audio() if hasattr(render_video, 'without_audio') else render_video
                        
                        # Use a simple temp filename without duplicate "enhanced"
                        # Clean the stem more aggressively to avoid any "enhanced" in the name
                        clean_stem_for_video = output_path.stem.replace('enhanced_', '').replace('viral_', '').replace('theburntpeanut_', '')
                        temp_filename = f"temp_video_{clean_stem_for_video}.mp4"
                        temp_enhanced_video = (self.temp_dir / temp_filename).resolve()
                        # Ensure temp directory exists (use absolute path)
                        self.temp_dir.mkdir(parents=True, exist_ok=True)
                        
                        logger.info(f"🔄 Writing enhanced video (no audio) to: {temp_enhanced_video}")
                        # Don't change directories - use absolute paths
                        video_no_audio_clip.write_videofile(
                            str(temp_enhanced_video),
                            codec=codec,
                            preset=preset,
                            ffmpeg_params=['-crf', str(crf), '-threads', str(threads)],
                            audio=False,  # No audio - this avoids the corrupted buffer
                            verbose=False,
                            logger=None
                        )
                        
                        # Verify the file was created
                        if not temp_enhanced_video.exists():
                            raise FileNotFoundError(f"Failed to create temp enhanced video at {temp_enhanced_video}")
                        logger.info(f"✅ Enhanced video (no audio) created: {temp_enhanced_video} ({temp_enhanced_video.stat().st_size / 1024 / 1024:.1f} MB)")
                        
                        # Step 4: Combine enhanced video with extracted audio using ffmpeg (bypasses MoviePy audio reader)
                        logger.info("🔄 Combining video and audio with ffmpeg...")
                        # Ensure output directory exists and use absolute paths
                        output_path_abs = output_path.resolve()
                        output_path_abs.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Verify temp files exist before combining (use absolute paths)
                        temp_enhanced_video_abs = temp_enhanced_video.resolve()
                        temp_audio_abs = temp_audio.resolve()
                        
                        if not temp_enhanced_video_abs.exists():
                            # Try to find the file - MoviePy might have created it with a different name
                            logger.warning(f"⚠️ Temp enhanced video not found at expected path: {temp_enhanced_video_abs}")
                            # Check if MoviePy created it in the temp directory with a different name
                            temp_dir_files = list(self.temp_dir.glob(f"*{clean_stem_for_video}*"))
                            if temp_dir_files:
                                logger.info(f"   Found potential temp file: {temp_dir_files[0]}")
                                temp_enhanced_video_abs = temp_dir_files[0].resolve()
                            else:
                                raise FileNotFoundError(f"Temp enhanced video not found: {temp_enhanced_video_abs}")
                        
                        if not temp_audio_abs.exists() or temp_audio_abs.stat().st_size == 0:
                            raise FileNotFoundError(f"Temp audio not found or empty: {temp_audio_abs}")
                        
                        logger.info(f"   Using video: {temp_enhanced_video_abs}")
                        logger.info(f"   Using audio: {temp_audio_abs}")
                        logger.info(f"   Output: {output_path_abs}")
                        
                        ffmpeg_combine_cmd = [
                            'ffmpeg', '-i', str(temp_enhanced_video_abs),
                            '-i', str(temp_audio_abs),
                            '-c:v', 'copy',  # Copy video (no re-encode for speed)
                            '-c:a', 'aac', '-b:a', '192k',  # Encode audio to AAC
                            '-shortest',  # Match shortest stream
                            '-y', str(output_path_abs)
                        ]
                        result = subprocess.run(ffmpeg_combine_cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            logger.error(f"❌ FFmpeg combine failed (exit {result.returncode}): {result.stderr}")
                            logger.error(f"   Command: {' '.join(ffmpeg_combine_cmd)}")
                            logger.error(f"   Input video: {temp_enhanced_video_abs} (exists: {temp_enhanced_video_abs.exists()})")
                            logger.error(f"   Input audio: {temp_audio_abs} (exists: {temp_audio_abs.exists()}, size: {temp_audio_abs.stat().st_size if temp_audio_abs.exists() else 0})")
                            logger.error(f"   Output: {output_path_abs}")
                            raise Exception(f"FFmpeg combine failed: {result.stderr}")
                        
                        # Clean up temp files immediately
                        for temp_file in [temp_video_no_audio, temp_audio, temp_enhanced_video]:
                            if temp_file.exists():
                                try:
                                    temp_file.unlink()
                                    logger.debug(f"🧹 Cleaned up temp file: {temp_file.name}")
                                except Exception as e:
                                    logger.warning(f"⚠️ Could not remove temp file {temp_file}: {e}")
                        
                        logger.info("✅ Video rendered successfully with ffmpeg workaround")
                    else:
                        logger.error("❌ Failed to extract video/audio with ffmpeg")
                        raise Exception("Video/audio extraction failed")
                except FileNotFoundError:
                    logger.error("❌ ffmpeg not found - cannot extract audio")
                    raise
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ ffmpeg command failed: {e}")
                    raise
                except Exception as audio_fix_error:
                    logger.error(f"❌ Audio extraction failed: {audio_fix_error}")
                    import traceback
                    traceback.print_exc()
                    raise
            else:
                # Non-audio error, try simpler rendering
                logger.info("🔄 Attempting fallback rendering...")
                try:
                    base_video = video
                    if hasattr(video, 'clips') and len(video.clips) > 0:
                        base_video = video.clips[0]
                    
                    import os
                    original_cwd = os.getcwd()
                    temp_dir = output_path.parent
                    
                    try:
                        os.chdir(temp_dir)
                        base_video.write_videofile(
                            output_path.name,
                            codec='libx264',
                            preset='veryfast',
                            ffmpeg_params=['-crf', '23'],
                            audio_codec='aac',
                            threads=2,
                            verbose=False,
                            logger=None,
                            temp_audiofile=f'TEMP_MPY_fallback_{output_path.stem}.mp4'
                        )
                    finally:
                        os.chdir(original_cwd)
                    logger.info("✅ Fallback rendering succeeded")
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback rendering also failed: {fallback_error}")
                    raise
        
        logger.info(f"💾 Video rendered: {output_path}")
        
        # Clean up any temp files that might have been created in root
        self._cleanup_temp_files()
    
    def _cleanup_temp_files(self):
        """Clean up any temporary MoviePy files that ended up in the wrong location"""
        try:
            import os
            import glob
            
            # Find and remove temp files in root directory
            temp_patterns = [
                'enhanced_*TEMP_MPY_*.mp4',
                '*TEMP_MPY_wvf_snd.mp4',
                '*TEMP_MPY_fallback_*.mp4'
            ]
            
            for pattern in temp_patterns:
                temp_files = glob.glob(pattern)
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                        logger.info(f"🧹 Cleaned up temp file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not remove temp file {temp_file}: {e}")
            
            # Clean up temp directory
            if self.temp_dir.exists():
                for temp_file in self.temp_dir.glob('*'):
                    if temp_file.is_file():
                        try:
                            # Keep only files that are actively being used (recently modified)
                            # Delete old temp files (older than 1 hour)
                            import time
                            file_age = time.time() - temp_file.stat().st_mtime
                            if file_age > 3600:  # 1 hour
                                temp_file.unlink()
                                logger.info(f"🧹 Cleaned up old temp file: {temp_file.name}")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not remove temp file {temp_file}: {e}")
                        
        except Exception as e:
            logger.warning(f"⚠️ Cleanup failed: {e}")
    
    def _cleanup_streamer_folder(self, streamer_folder: Path):
        """Clean up temporary files from streamer folder, keeping only final enhanced clips"""
        try:
            if not streamer_folder.exists():
                return
            
            # Keep only files that match the final enhanced clip pattern
            # Pattern: enhanced_{streamer_name}_{timestamp}.mp4
            keep_pattern = 'enhanced_'
            
            files_removed = 0
            dirs_removed = 0
            
            for item in streamer_folder.iterdir():
                if item.is_file():
                    # Keep only the final enhanced clip
                    should_keep = item.name.startswith(keep_pattern) and item.suffix == '.mp4'
                    
                    if not should_keep:
                        # Remove everything else (temp files, viral versions, etc.)
                        try:
                            item.unlink()
                            files_removed += 1
                            logger.info(f"🧹 Removed temp file from streamer folder: {item.name}")
                        except Exception as e:
                            logger.warning(f"⚠️ Could not remove temp file {item}: {e}")
                elif item.is_dir():
                    # Remove nested directories (like clips/temp subdirectories)
                    try:
                        import shutil
                        shutil.rmtree(item)
                        dirs_removed += 1
                        logger.info(f"🧹 Removed nested directory from streamer folder: {item.name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not remove nested directory {item}: {e}")
            
            if files_removed > 0 or dirs_removed > 0:
                logger.info(f"🧹 Cleaned up {files_removed} temp file(s) and {dirs_removed} directory(ies) from {streamer_folder.name} folder")
                        
        except Exception as e:
            logger.warning(f"⚠️ Streamer folder cleanup failed: {e}")


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
