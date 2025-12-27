#!/usr/bin/env python3
"""
Twitch Clip Bot - Phase 1 Complete System
Automatically creates clips when metrics spike and uploads to TikTok
"""

import asyncio
import json
import logging
import os
import sys
import time
import threading
import queue
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from pathlib import Path

import click
import requests
import websockets
import yaml
from dotenv import load_dotenv

# Fix encoding for Windows console (handle emoji output)
if sys.platform == 'win32':
    try:
        # Try to set UTF-8 encoding
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        else:
            # Fallback for older Python versions
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')
    except Exception:
        # If encoding setup fails, continue anyway
        pass

# Load environment variables for sensitive credentials
load_dotenv()

# Import our enhanced modules
try:
    from clip_enhancer_v2 import ClipEnhancerV2 as ClipEnhancer, check_dependencies as check_deps_v1
    from clip_enhancer_v2 import ClipEnhancerV2, check_dependencies
    from viral_detector import ViralDetector
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    ClipEnhancer = None
    ClipEnhancerV2 = None
    check_dependencies = None
    check_deps_v1 = None
    ViralDetector = None

# Suppress TensorFlow warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')  # Suppress TF INFO messages

# Configure logging with cleaner format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class TwitchClipBot:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize bot with YAML configuration"""
        self.config_path = Path(config_path)
        self.load_config()
        
        # Initialize components
        self.clip_enhancer = ClipEnhancer() if ClipEnhancer else None
        self.clip_enhancer_v2 = ClipEnhancerV2(config_path) if ClipEnhancerV2 else None
        self.viral_detector = ViralDetector(self.config) if ViralDetector else None
        
        # Twitch API setup from environment variables (for security)
        self.client_id = os.getenv('TWITCH_CLIENT_ID')
        self.client_secret = os.getenv('TWITCH_CLIENT_SECRET')
        self.oauth_token = os.getenv('TWITCH_OAUTH_TOKEN')
        
        if not all([self.client_id, self.client_secret]):
            raise ValueError("Missing Twitch credentials in .env file")
        
        self.access_token = None
        self.headers = {}
        
        # Monitoring data
        self.chat_messages = deque(maxlen=1000)  # Store recent chat messages with timestamps
        self.viewer_history = deque(maxlen=60)   # Store viewer counts for last 60 checks
        self.last_clip_time = 0
        
        # Current streamer being monitored
        self.current_streamer = None
        self.clip_cooldown = 60  # Minimum seconds between clips
        
        # Clip enhancement
        self.enhancer = ClipEnhancer() if ClipEnhancer else None
        self.auto_enhance = self.config.get('global', {}).get('auto_enhance', True)
        
        # Async enhancement queue to prevent blocking monitoring
        self.enhancement_queue = queue.Queue()
        self.enhancement_worker = None
        self.enhancement_running = False
        
        # Always-on mode integration
        self.always_on_mode = False
        self.controller_enhancement_queue = None
        self.enhancement_queue_file = None  # File-based queue communication
    
    def start_enhancement_worker(self):
        """Start the background enhancement worker thread"""
        if not self.enhancement_running:
            self.enhancement_running = True
            self.enhancement_worker = threading.Thread(target=self._enhancement_worker, daemon=True)
            self.enhancement_worker.start()
            logger.info("🔧 Started enhancement worker thread")
    
    def stop_enhancement_worker(self):
        """Stop the background enhancement worker thread"""
        if self.enhancement_running:
            self.enhancement_running = False
            # Add poison pill to wake up worker
            self.enhancement_queue.put(None)
            if self.enhancement_worker:
                self.enhancement_worker.join(timeout=2)
            logger.info("🛑 Stopped enhancement worker thread")
    
    def _enhancement_worker(self):
        """Background worker that processes enhancement queue"""
        logger.info("⚡ Enhancement worker ready")
        while self.enhancement_running:
            try:
                # Get next enhancement job (block for up to 1 second)
                job = self.enhancement_queue.get(timeout=1)
                
                # Poison pill to stop worker
                if job is None:
                    break
                
                clip_url, reason = job
                logger.info(f"🎨 Processing enhancement job: {clip_url}")
                
                self._process_enhancement(clip_url, reason)
                
                self.enhancement_queue.task_done()
                
            except queue.Empty:
                continue  # Timeout, check if still running
            except Exception as e:
                logger.error(f"❌ Enhancement worker error: {e}")
        
        logger.info("🏁 Enhancement worker stopped")
    
    def _send_clip_to_controller(self, clip_url: str, reason: str):
        """Send clip to Celery/Redis queue for enhancement"""
        try:
            from datetime import datetime
            from celery_tasks import enhance_clip_task
            
            # Create clip job data
            # Get current game name from stream data
            current_game = 'Unknown'
            try:
                stream_data = self.get_stream_data()
                if stream_data:
                    current_game = stream_data.get('game_name', 'Unknown')
            except Exception:
                pass
            
            job_data = {
                'clip_url': clip_url,
                'streamer_name': self.current_streamer.get('name'),
                'streamer_handle': self.current_streamer.get('twitch_username'),
                'game_name': current_game,
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'streamer_config': self.current_streamer
            }
            
            # Send to Celery/Redis queue (async, non-blocking)
            task = enhance_clip_task.delay(job_data)
            
            logger.info(f"📤 Sent clip to Celery queue (task ID: {task.id}): {clip_url}")
            logger.info(f"🎯 Reason: {reason}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send clip to Celery: {e}")
            logger.warning("⚠️ Make sure Redis and Celery worker are running!")
    
    def _process_enhancement(self, clip_url: str, reason: str):
        """Process a single enhancement job (standalone mode only)"""
        try:
            if self.clip_enhancer_v2:
                # Download clip using v2 enhancer
                clip_path = self.clip_enhancer_v2.download_clip(clip_url)
                if clip_path:
                    logger.info(f"📥 Downloaded clip for enhancement")
                    
                    # Enhance with v2 system using current streamer config
                    enhanced_path, telemetry = self.clip_enhancer_v2.enhance_clip(clip_path, self.current_streamer)
                    
                    logger.info(f"✨ Enhanced clip saved: {enhanced_path}")
                    logger.info(f"📊 Processing time: {telemetry.processing_time_ms}ms")
                    logger.info(f"📊 Words rendered: {telemetry.words_rendered}")
                    logger.info(f"📊 Emphasis hits: {telemetry.emphasis_hits}")
                else:
                    logger.warning("Failed to download clip for enhancement")
            else:
                logger.warning("Enhancement v2 not available")
        except Exception as e:
            logger.error(f"Enhancement failed for {clip_url}: {e}")
            import traceback
            traceback.print_exc()

    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("✅ Configuration loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            self.config = {}
    
    def get_enabled_streamers(self) -> List[Dict]:
        """Get list of enabled streamers from config"""
        streamers = self.config.get('streamers', [])
        return [s for s in streamers if s.get('enabled', False)]
    
    def get_streamer_config(self, streamer_name: str) -> Optional[Dict]:
        """Get configuration for a specific streamer"""
        for streamer in self.config.get('streamers', []):
            if streamer.get('name') == streamer_name:
                return streamer
        return None
    
    def set_current_streamer(self, streamer_config: Dict):
        """Set the current streamer and initialize viral detector"""
        self.current_streamer = streamer_config
        
        # Initialize viral detector for this streamer if using advanced algorithm
        if self.viral_detector:
            self.viral_detector.configure_for_streamer(streamer_config.get('name', ''))
            self.viral_detector.set_stream_status(True)  # Assume we're starting monitoring because stream is live
            logger.info(f"🧪 Viral detector configured for {streamer_config.get('name', 'unknown')} | Stream start time set: {self.viral_detector.stream_start_time}")
        
    async def authenticate(self) -> bool:
        """Get OAuth access token from Twitch API"""
        try:
            # Always get client credentials for general API calls (stream data, etc.)
            url = "https://id.twitch.tv/oauth2/token"
            params = {
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'grant_type': 'client_credentials'
            }
            
            response = requests.post(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data['access_token']
            
            # Use client credentials for general API calls
            self.headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Client-Id': self.client_id,
                'Content-Type': 'application/json'
            }
            
            # Store OAuth token separately for clip creation (if provided)
            if self.oauth_token:
                logger.info("Using client credentials for API calls, OAuth token available for clip creation")
            else:
                logger.info("Using client credentials (limited clip permissions - need OAuth token for clip creation)")
            
            logger.info("Successfully authenticated with Twitch API")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def get_stream_data(self, retry_on_timeout: bool = True) -> Optional[Dict]:
        """Get current stream data including viewer count with retry logic for timeouts"""
        if not self.current_streamer:
            logger.error("No current streamer set")
            return None
        
        import time
        broadcaster_id = self.current_streamer.get('broadcaster_id')
        max_retries = 3 if retry_on_timeout else 1
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Add cache-busting parameter to get fresh data
                url = f"https://api.twitch.tv/helix/streams?user_id={broadcaster_id}&_t={int(time.time() * 1000)}"
                # Add cache-control headers to ensure fresh data
                fresh_headers = self.headers.copy()
                fresh_headers['Cache-Control'] = 'no-cache'
                fresh_headers['Pragma'] = 'no-cache'
                response = requests.get(url, headers=fresh_headers, timeout=15)  # Increased timeout to 15s
                response.raise_for_status()
                
                data = response.json()
                if data['data']:
                    stream_info = data['data'][0]
                    # Log more details about the stream data
                    viewer_count = stream_info['viewer_count']
                    game_name = stream_info.get('game_name', 'Unknown')
                    started_at = stream_info.get('started_at', 'Unknown')
                    logger.debug(f"Fresh stream data: {viewer_count} viewers, {game_name}, started: {started_at}")
                    return stream_info
                else:
                    logger.info("Stream is offline")
                    return None
                    
            except requests.Timeout as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ API timeout (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"❌ API timeout after {max_retries} attempts: {e}")
                    # Return a special marker to indicate timeout (not offline)
                    return {"_timeout": True}
                    
            except requests.RequestException as e:
                error_str = str(e)
                # Distinguish between network errors and actual API errors
                if "Failed to resolve" in error_str or "Max retries exceeded" in error_str:
                    logger.error(f"❌ Network error: {e}")
                    return None  # Treat as offline
                elif "timeout" in error_str.lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Request timeout (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"❌ Request timeout after {max_retries} attempts: {e}")
                        return {"_timeout": True}
                else:
                    logger.error(f"❌ API error: {e}")
                    return None
        
        return None
    
    async def monitor_chat(self, broadcaster_name: str):
        """Monitor Twitch IRC chat for message activity"""
        uri = "wss://irc-ws.chat.twitch.tv:443"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Send authentication
                await websocket.send("PASS oauth:anonymous")
                await websocket.send("NICK justinfan12345")  # Anonymous connection
                await websocket.send(f"JOIN #{broadcaster_name}")
                
                logger.info(f"💬 Chat connected: #{broadcaster_name}")
                
                async for message in websocket:
                    if message.startswith("PING"):
                        await websocket.send("PONG :tmi.twitch.tv")
                        continue
                    
                    if "PRIVMSG" in message:
                        # Record chat message timestamp
                        current_time = time.time()
                        self.chat_messages.append(current_time)
                        
                        # Parse username and message for logging and viral detector
                        try:
                            parts = message.split(":", 2)
                            if len(parts) >= 3:
                                user_info = parts[1].split("!")[0]
                                chat_message = parts[2].strip()
                                logger.debug(f"Chat: {user_info}: {chat_message}")
                                
                                # Feed to viral detector if using advanced algorithm
                                if self.viral_detector:
                                    user_id = f"user_{hash(user_info) % 1000000}"
                                    self.viral_detector.add_chat_message(
                                        username=user_info,
                                        user_id=user_id,
                                        message=chat_message
                                    )
                        except Exception as e:
                            logger.debug(f'Error parsing chat: {e}')
                            
        except Exception as e:
            logger.error(f"Chat monitoring error: {e}")
    
    def get_broadcaster_name(self) -> Optional[str]:
        """Get broadcaster username from config or API"""
        if not self.current_streamer:
            return None
        
        # First try to use the twitch_username from config (avoids API call)
        twitch_username = self.current_streamer.get('twitch_username')
        if twitch_username:
            return twitch_username
        
        # Fallback to API lookup if twitch_username not in config
        try:
            broadcaster_id = self.current_streamer.get('broadcaster_id')
            url = f"https://api.twitch.tv/helix/users?id={broadcaster_id}"
            # Use public headers if available, otherwise fall back to self.headers
            headers = getattr(self, 'public_headers', self.headers)
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if data['data']:
                return data['data'][0]['login']
            return None
            
        except requests.RequestException as e:
            logger.error(f"Failed to get broadcaster name: {e}")
            return None
    
    def calculate_chat_mps(self, window_seconds: int = 10) -> float:
        """Calculate messages per second in the given time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Count messages in the time window
        recent_messages = [msg_time for msg_time in self.chat_messages if msg_time > cutoff_time]
        mps = len(recent_messages) / window_seconds
        
        return mps
    
    def calculate_viewer_change(self) -> Tuple[float, int]:
        """Calculate viewer count percentage change over the last minute"""
        if len(self.viewer_history) < 2:
            return 0.0, 0
        
        current_viewers = self.viewer_history[-1]
        past_viewers = self.viewer_history[0] if len(self.viewer_history) >= 60 else self.viewer_history[0]
        
        if past_viewers == 0:
            return 0.0, current_viewers
        
        percentage_change = ((current_viewers - past_viewers) / past_viewers) * 100
        return percentage_change, current_viewers
    
    def detect_spikes(self) -> Tuple[bool, str]:
        """Detect if metrics indicate a clip-worthy moment"""
        if not self.current_streamer:
            return False, "No current streamer configured"
        
        # Check if using advanced viral detection algorithm
        use_viral_alg = self.config.get('global', {}).get('use_test_virality_alg', False)
        
        if use_viral_alg and self.viral_detector:
            # Use advanced viral detection algorithm
            should_clip, reason, breakdown = self.viral_detector.should_create_clip()
            
            # DEBUG LOGGING - Remove after testing
            logger.info(f"🧪 VIRAL CHECK | Result: {should_clip} | {reason}")
            
            if should_clip:
                # Log detailed breakdown for analysis
                components = breakdown.get('components', {})
                logger.info(f"🔥 Viral algorithm breakdown:")
                logger.info(f"   Total score: {breakdown.get('total_score', 0):.3f}")
                logger.info(f"   Chat component: {components.get('chat', 0):.3f} (MPS: {breakdown.get('current_mps', 0):.1f})")
                logger.info(f"   Viewer component: {components.get('viewer', 0):.3f} (Delta: {breakdown.get('viewer_delta_percent', 0):.1f}%)")
                logger.info(f"   Engagement component: {components.get('events', 0):.3f}")
                logger.info(f"   Follow component: {components.get('follow', 0):.3f}")
                logger.info(f"   Keyword component: {components.get('keyword', 0):.3f}")
                logger.info(f"   Sentiment component: {components.get('sentiment', 0):.3f}")
                logger.info(f"   Emote component: {components.get('emote', 0):.3f}")
            
            return should_clip, reason
        else:
            # Use legacy threshold-based detection
            return self._detect_spikes_legacy()
    
    def _detect_spikes_legacy(self) -> Tuple[bool, str]:
        """Legacy threshold-based spike detection"""
        # Get thresholds from current streamer config or global defaults
        default_thresholds = self.config.get('global', {}).get('default_thresholds', {})
        streamer_thresholds = self.current_streamer.get('thresholds', {})
        
        chat_threshold = streamer_thresholds.get('chat_spike', default_thresholds.get('chat_spike', 30))
        viewer_threshold = streamer_thresholds.get('viewer_increase', default_thresholds.get('viewer_increase', 20))
        time_window = streamer_thresholds.get('time_window', default_thresholds.get('time_window', 10))
        
        # Check chat activity spike
        chat_mps = self.calculate_chat_mps(time_window)
        if chat_mps > chat_threshold:
            return True, f"Chat spike detected: {chat_mps:.1f} messages/second (threshold: {chat_threshold})"
        
        # Check viewer count spike
        viewer_change, current_viewers = self.calculate_viewer_change()
        if viewer_change > viewer_threshold:
            return True, f"Viewer spike detected: +{viewer_change:.1f}% ({current_viewers} viewers, threshold: {viewer_threshold}%)"
        
        return False, ""
    
    def create_clip(self, reason: str = "") -> Optional[str]:
        """Create a clip and return the clip URL"""
        try:
            # Check cooldown
            current_time = time.time()
            if current_time - self.last_clip_time < self.clip_cooldown:
                logger.info(f"Clip creation on cooldown ({self.clip_cooldown}s)")
                return None
            
            broadcaster_id = self.current_streamer.get('broadcaster_id') if self.current_streamer else None
            if not broadcaster_id:
                logger.error("No broadcaster ID available for clip creation")
                return None
                
            url = f"https://api.twitch.tv/helix/clips?broadcaster_id={broadcaster_id}"
            
            # Must use OAuth token for clip creation (client credentials don't have clips:edit permission)
            if not self.oauth_token:
                logger.error("❌ Cannot create clip: Missing OAuth token with clips:edit permission")
                logger.error("   Run: python clipppy.py oauth-help --generate-url")
                return None
            
            # Debug: Log token info (first 10 chars only for security)
            token_preview = self.oauth_token[:10] + "..." if len(self.oauth_token) > 10 else self.oauth_token
            logger.info(f"🔑 Using OAuth token: {token_preview} (length: {len(self.oauth_token)})")
            
            # Validate token first by checking its info
            try:
                validate_url = "https://id.twitch.tv/oauth2/validate"
                validate_response = requests.get(validate_url, headers={'Authorization': f'OAuth {self.oauth_token}'})
                if validate_response.status_code == 200:
                    token_info = validate_response.json()
                    logger.info(f"✅ Token validated - Scopes: {token_info.get('scopes', [])}")
                    if 'clips:edit' not in token_info.get('scopes', []):
                        logger.error("❌ Token missing 'clips:edit' scope!")
                        logger.error("   Get a new token with clips:edit scope: python clipppy.py oauth-help --generate-url")
                        return None
                else:
                    logger.warning(f"⚠️ Token validation failed: {validate_response.status_code}")
                    logger.warning(f"   Response: {validate_response.text}")
                    logger.warning("   Token may be expired or invalid. Get a new token: python clipppy.py oauth-help --generate-url")
            except Exception as e:
                logger.warning(f"⚠️ Could not validate token: {e}")
            
            clip_headers = {
                'Authorization': f'Bearer {self.oauth_token}',
                'Client-Id': self.client_id,
                'Content-Type': 'application/json'
            }
            logger.info(f"📤 Creating clip for broadcaster {broadcaster_id}...")
            response = requests.post(url, headers=clip_headers)
            response.raise_for_status()
            
            data = response.json()
            if data['data']:
                clip_id = data['data'][0]['id']
                edit_url = data['data'][0]['edit_url']
                
                # Convert edit URL to view URL
                clip_url = edit_url.replace('/manager', '')
                
                self.last_clip_time = current_time
                logger.info(f"✅ Clip created: {clip_url}")
                if reason:
                    logger.info(f"   Reason: {reason}")
                
                # Auto-enhance clip if enabled - add to queue for non-blocking processing
                if self.auto_enhance and self.clip_enhancer_v2:
                    try:
                        if self.always_on_mode:
                            # In always-on mode, send clip to central controller queue
                            self._send_clip_to_controller(clip_url, reason)
                        else:
                            # Standalone mode, process locally
                            queue_size = self.enhancement_queue.qsize()
                            self.enhancement_queue.put((clip_url, reason))
                            logger.info(f"📋 Added clip to local enhancement queue (position {queue_size + 1})")
                        
                        # Start worker if not running (only in standalone mode)
                        if not self.enhancement_running and not self.always_on_mode:
                            self.start_enhancement_worker()
                            
                    except Exception as e:
                        logger.error(f"Failed to queue enhancement: {e}")
                
                return clip_url
            
        except requests.RequestException as e:
            # Check for specific HTTP errors
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                if status_code == 429:
                    logger.warning("⚠️ Rate limited by Twitch API - waiting before retry")
                elif status_code == 401:
                    logger.error("❌ Unauthorized: Need user OAuth token with 'clips:edit' scope for clip creation")
                    logger.error("   Client credentials cannot create clips. You need to get a user OAuth token.")
                    logger.error("   Run: python clipppy.py oauth-help --generate-url")
                    logger.error("   Then add TWITCH_OAUTH_TOKEN=your_token to your .env file")
                else:
                    logger.error(f"❌ Failed to create clip: HTTP {status_code} - {e}")
            else:
                logger.error(f"❌ Failed to create clip: {e}")
        
        return None
    
    async def start_monitoring(self):
        """Start monitoring the stream for clip opportunities"""
        if not await self.authenticate():
            return
        
        broadcaster_name = self.get_broadcaster_name()
        if not broadcaster_name:
            logger.error("Could not get broadcaster username")
            return
        
        broadcaster_id = self.current_streamer.get('broadcaster_id')
        logger.info(f"Starting monitoring for {broadcaster_name} (ID: {broadcaster_id})")
        
        # Start enhancement worker thread
        self.start_enhancement_worker()
        
        # Start chat monitoring in background
        chat_task = asyncio.create_task(self.monitor_chat(broadcaster_name))
        
        # Track consecutive failures to distinguish timeout from offline
        consecutive_failures = 0
        max_consecutive_failures = 3  # Allow 3 failures before assuming offline
        
        try:
            while True:
                # Get stream data
                stream_data = self.get_stream_data()
                
                # Check if it's a timeout (not offline)
                if stream_data and stream_data.get("_timeout"):
                    consecutive_failures += 1
                    logger.warning(f"⚠️ API timeout ({consecutive_failures}/{max_consecutive_failures}) - continuing monitoring...")
                    await asyncio.sleep(10)  # Wait a bit before retrying
                    continue
                
                if stream_data is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        if self.always_on_mode:
                            logger.info(f"⚫ Stream appears offline after {consecutive_failures} consecutive failures - shutting down listener")
                            break  # Exit the loop, let controller handle restart when live
                        else:
                            logger.info("Stream is offline, waiting...")
                            await asyncio.sleep(30)
                            continue
                    else:
                        logger.warning(f"⚠️ Stream data unavailable ({consecutive_failures}/{max_consecutive_failures}) - retrying...")
                        await asyncio.sleep(10)
                        continue
                
                # Success - reset failure counter
                if consecutive_failures > 0:
                    logger.info(f"✅ Stream data recovered after {consecutive_failures} failures")
                consecutive_failures = 0
                
                # Record viewer count
                viewer_count = stream_data['viewer_count']
                self.viewer_history.append(viewer_count)
                
                # Feed viewer data to viral detector if using advanced algorithm
                if self.viral_detector:
                    self.viral_detector.add_viewer_data(viewer_count)
                
                # Check for spikes
                should_clip, reason = self.detect_spikes()
                
                if should_clip:
                    logger.info(f"🚨 Spike detected! {reason}")
                    self.create_clip(reason)
                
                # Log current stats with clean formatting
                chat_mps = self.calculate_chat_mps(10)
                viewer_change, _ = self.calculate_viewer_change()
                
                # Only log the essential heartbeat info, cleanly formatted
                timestamp = datetime.now().strftime("%H:%M:%S")
                logger.info(f"🔴 [{timestamp}] {self.current_streamer['name'].upper()} | Viewers: {viewer_count:,} | Chat: {chat_mps:.1f}/s | Change: {viewer_change:+.1f}%")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            chat_task.cancel()
            # Stop enhancement worker
            self.stop_enhancement_worker()
    
    async def start_monitoring_streamer(self, streamer_config: Dict):
        """Start monitoring a specific streamer"""
        # Set the current streamer
        self.set_current_streamer(streamer_config)
        
        # Start monitoring
        await self.start_monitoring()
    
    async def start_monitoring_all(self):
        """Start monitoring all enabled streamers"""
        enabled_streamers = self.get_enabled_streamers()
        
        if len(enabled_streamers) == 1:
            # If only one streamer, monitor directly
            await self.start_monitoring_streamer(enabled_streamers[0])
        else:
            # For multiple streamers, we'd need concurrent monitoring
            # For now, just monitor the first enabled streamer
            logger.warning(f"⚠️ Multiple streamers detected ({len(enabled_streamers)}), monitoring first one only")
            logger.info("💡 Full multi-streamer support coming soon!")
            await self.start_monitoring_streamer(enabled_streamers[0])


# CLI Interface
@click.group()
def cli():
    """Twitch Clip Bot - Automatically create clips when metrics spike"""
    pass


@cli.command()
@click.option('--streamer', help='Specific streamer to monitor (default: all enabled)')
@click.option('--always-on-mode', is_flag=True, help='Run in always-on mode (managed by controller)')
def start(streamer, always_on_mode):
    """Start monitoring streamers for clip opportunities"""
    try:
        bot = TwitchClipBot()
        
        if streamer:
            # Monitor specific streamer
            streamer_config = bot.get_streamer_config(streamer)
            if not streamer_config:
                click.echo(f"❌ Streamer '{streamer}' not found in config")
                return
            if not streamer_config.get('enabled', False):
                click.echo(f"❌ Streamer '{streamer}' is disabled in config")
                return
            
            if always_on_mode:
                try:
                    click.echo(f"🤖 Starting always-on listener for {streamer}")
                except UnicodeEncodeError:
                    click.echo(f"Starting always-on listener for {streamer}")
                bot.always_on_mode = True
            else:
                click.echo(f"🎯 Starting standalone monitoring for {streamer}")
                bot.always_on_mode = False
            
            asyncio.run(bot.start_monitoring_streamer(streamer_config))
        else:
            # Monitor all enabled streamers
            enabled_streamers = bot.get_enabled_streamers()
            if not enabled_streamers:
                click.echo("❌ No enabled streamers found in config")
                return
            
            click.echo(f"🚀 Starting monitoring for {len(enabled_streamers)} streamers...")
            asyncio.run(bot.start_monitoring_all())
            
    except ValueError as e:
        try:
            click.echo(f"❌ Configuration error: {e}")
        except UnicodeEncodeError:
            click.echo(f"ERROR: Configuration error: {e}")
        click.echo("Please check your config.yaml file.")
    except Exception as e:
        try:
            click.echo(f"❌ Error: {e}")
        except UnicodeEncodeError:
            click.echo(f"ERROR: {e}")


@cli.command()
@click.option('--streamer', default='theburntpeanut', help='Streamer to create clip for')
@click.option('--enhance', is_flag=True, help='Also enhance the clip after creation')
def testclip(streamer, enhance):
    """Create a test clip manually (force clip creation)"""
    try:
        bot = TwitchClipBot()
        
        async def test():
            if await bot.authenticate():
                # Set the streamer first
                streamer_config = bot.get_streamer_config(streamer)
                if not streamer_config:
                    click.echo(f"❌ Streamer '{streamer}' not found in config")
                    return
                
                bot.set_current_streamer(streamer_config)
                click.echo(f"🎯 Testing clip creation for {streamer}...")
                
                # Force create clip (bypasses all checks)
                clip_url = bot.create_clip("Manual test clip (forced)")
                if clip_url:
                    click.echo(f"✅ Test clip created successfully: {clip_url}")
                    
                    if enhance:
                        click.echo(f"\n🎬 Starting full enhancement pipeline...")
                        # Download and enhance using v2 enhancer
                        if bot.clip_enhancer_v2:
                            click.echo(f"📥 Downloading clip...")
                            clip_path = bot.clip_enhancer_v2.download_clip(clip_url)
                            if clip_path:
                                click.echo(f"✅ Downloaded to: {clip_path}")
                                click.echo(f"🎨 Enhancing clip...")
                                enhanced_path, telemetry = bot.clip_enhancer_v2.enhance_clip(clip_path, streamer_config)
                                click.echo(f"✅ Enhanced clip saved: {enhanced_path}")
                                click.echo(f"📊 Processing time: {telemetry.processing_time_ms}ms")
                                click.echo(f"📊 Words rendered: {telemetry.words_rendered}")
                                click.echo(f"📊 Emphasis hits: {telemetry.emphasis_hits}")
                            else:
                                click.echo("❌ Failed to download clip")
                        else:
                            click.echo("❌ Enhancement v2 not available")
                else:
                    click.echo("❌ Failed to create test clip - check OAuth token")
            else:
                click.echo("❌ Authentication failed")
        
        asyncio.run(test())
        
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        click.echo("Please check your .env file has all required variables.")
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
def config():
    """Show current configuration status"""
    try:
        bot = TwitchClipBot()
        click.echo("🔧 Configuration Status:")
        click.echo(f"   Twitch Client ID: {'✅ Set' if bot.client_id else '❌ Missing'}")
        click.echo(f"   Twitch Client Secret: {'✅ Set' if bot.client_secret else '❌ Missing'}")
        click.echo(f"   Twitch OAuth Token: {'✅ Set' if bot.oauth_token else '❌ Missing (needed for clips)'}")
        
        # Show streamers
        enabled_streamers = bot.get_enabled_streamers()
        disabled_streamers = [s for s in bot.config.get('streamers', []) if not s.get('enabled', False)]
        
        click.echo(f"\n📺 Streamers:")
        click.echo(f"   Enabled: {len(enabled_streamers)}")
        for streamer in enabled_streamers:
            tiktok_user = streamer.get('tiktok_account', {}).get('username', 'Not set')
            click.echo(f"     🟢 {streamer['name']} → @{tiktok_user}")
        
        click.echo(f"   Disabled: {len(disabled_streamers)}")
        for streamer in disabled_streamers:
            click.echo(f"     🔴 {streamer['name']}")
        
        # Show enhancement status
        click.echo(f"🎬 Clip Enhancement: {'✅ Ready' if bot.clip_enhancer else '❌ Not available'}")
        
        if all([bot.client_id, bot.client_secret]):
            click.echo("\n🔑 Testing authentication...")
            
            async def test_auth():
                if await bot.authenticate():
                    click.echo("   ✅ Authentication successful")
                    
                    # Test with enabled streamers
                    enabled_streamers = bot.get_enabled_streamers()
                    if enabled_streamers:
                        for streamer in enabled_streamers[:1]:  # Test first enabled streamer
                            bot.current_streamer = streamer
                            broadcaster_name = bot.get_broadcaster_name()
                            if broadcaster_name:
                                click.echo(f"   ✅ Stream data accessible for {streamer['name']} (@{broadcaster_name})")
                            else:
                                click.echo(f"   ⚠️  Could not get stream data for {streamer['name']}")
                    else:
                        click.echo("   ⚠️  No enabled streamers to test")
                    
                    if not bot.oauth_token:
                        click.echo("\n⚠️  Note: You're using client credentials.")
                        click.echo("   For clip creation, you need a user OAuth token.")
                        click.echo("   Run: python clipppy.py oauth-help --generate-url")
                        click.echo("   Then add TWITCH_OAUTH_TOKEN=your_token to .env file")
                else:
                    click.echo("   ❌ Authentication failed")
            
            asyncio.run(test_auth())
    
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}")


@cli.command()
@click.option('--generate-url', is_flag=True, help='Generate OAuth URL for token')
def oauth_help(generate_url):
    """Help with getting OAuth token for clip creation"""
    try:
        bot = TwitchClipBot()
        
        if generate_url:
            # Generate OAuth URL
            oauth_url = (
                f"https://id.twitch.tv/oauth2/authorize"
                f"?client_id={bot.client_id}"
                f"&redirect_uri=http://localhost:8080"
                f"&response_type=token"
                f"&scope=clips:edit"
            )
            click.echo("🔗 OAuth URL generated:")
            click.echo(f"   {oauth_url}")
            click.echo("\n📝 Instructions:")
            click.echo("1. Open this URL in your browser")
            click.echo("2. Authorize the application")
            click.echo("3. Copy the 'access_token' from the redirect URL")
            click.echo("4. Add it to your .env file as TWITCH_OAUTH_TOKEN=your_token_here")
        else:
            click.echo("🎯 To create clips, you need a user OAuth token with 'clips:edit' scope.")
            click.echo("\n📚 Two options:")
            click.echo("1. Run: python clipppy.py oauth-help --generate-url")
            click.echo("2. Use a tool like https://twitchtokengenerator.com/ with 'clips:edit' scope")
            click.echo("\n⚙️  Once you have the token, add it to your .env file:")
            click.echo("   TWITCH_OAUTH_TOKEN=your_actual_token_here")
    
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}")


@cli.command()
@click.argument('clip_url')
@click.option('--text', help='Custom text overlay')
@click.option('--sound', help='Sound effect (airhorn, wow, epic, sus)')
@click.option('--position', default='center', help='Text position (center, top, bottom)')
@click.option('--no-captions', is_flag=True, help='Disable animated captions')
@click.option('--vertical', is_flag=True, help='Convert to vertical TikTok format')
def enhance(clip_url, text, sound, position, no_captions, vertical):
    """Enhance a Twitch clip with AI text and sound effects"""
    try:
        if not ClipEnhancer:
            click.echo("❌ Clip enhancement not available. Install dependencies:")
            click.echo("   pip install moviepy yt-dlp")
            if not check_dependencies():
                return
        
        enhancer = ClipEnhancer()
        
        click.echo(f"🎬 Enhancing clip: {clip_url}")
        
        # Auto-enhance or manual enhance
        if not text and not sound:
            # Auto-enhance
            result = enhancer.auto_enhance_clip(clip_url, "manual enhancement", use_captions=not no_captions)
        else:
            # Manual enhance
            video_path = enhancer.download_clip(clip_url)
            if not video_path:
                click.echo("❌ Failed to download clip")
                return
            
            if not text:
                text = enhancer.generate_viral_text("manual enhancement")
            
            enhanced_path = enhancer.enhance_clip(
                video_path,
                text_overlay=text,
                sound_effect=sound,
                text_position=position,
                use_captions=not no_captions,
                vertical_format=vertical
            )
            
            result = {
                'enhanced_path': enhanced_path,
                'text_overlay': text,
                'sound_effect': sound,
                'has_captions': not no_captions
            } if enhanced_path else None
        
        if result:
            click.echo(f"✨ Enhanced clip saved: {result['enhanced_path']}")
            if result.get('has_captions'):
                click.echo("   Features: Animated captions")
            if result.get('text_overlay'):
                click.echo(f"   Text: {result['text_overlay']}")
            if result.get('sound_effect'):
                click.echo(f"   Sound: {result['sound_effect']}")
        else:
            click.echo("❌ Enhancement failed")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.argument('clip_url')
@click.option('--horizontal', is_flag=True, help='Keep original aspect ratio (default: vertical)')
def caption(clip_url, horizontal):
    """Create MrBeast-style animated captions for a Twitch clip"""
    try:
        if not ClipEnhancer:
            click.echo("❌ Clip enhancement not available. Install dependencies:")
            click.echo("   pip install moviepy yt-dlp openai-whisper")
            if not check_dependencies():
                return
        
        enhancer = ClipEnhancer()
        
        click.echo(f"🎭 Creating captioned clip: {clip_url}")
        click.echo("   This may take a few minutes for transcription...")
        
        result = enhancer.create_captioned_clip(clip_url, vertical=not horizontal)
        
        if result:
            click.echo(f"✨ Captioned clip created: {result['enhanced_path']}")
            click.echo(f"   Format: {result['format']}")
            click.echo(f"   Style: {result['style']} (animated captions)")
            click.echo("   Features: Speech-to-text, viral text highlighting, bounce animations")
        else:
            click.echo("❌ Caption creation failed")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
def setup_enhancement():
    """Setup clip enhancement dependencies and check configuration"""
    click.echo("🔧 Setting up clip enhancement...")
    
    # Check dependencies
    if check_dependencies and check_dependencies():
        click.echo("✅ All dependencies installed")
    else:
        click.echo("❌ Missing dependencies. Install with:")
        click.echo("   pip install moviepy yt-dlp")


@cli.command()
@click.argument('clip_url')
@click.option('--streamer', default='jynxzi', help='Streamer name for preset selection')
def enhance_v2(clip_url, streamer):
    """Enhance clip with v2 preset-based system"""
    try:
        if not ClipEnhancerV2:
            click.echo("❌ Enhancement v2 not available")
            return
        
        bot = TwitchClipBot()
        streamer_config = bot.get_streamer_config(streamer)
        
        if not streamer_config:
            click.echo(f"❌ Streamer '{streamer}' not found in config")
            return
        
        # Use old enhancer to download clip
        if ClipEnhancer:
            enhancer_v1 = ClipEnhancer()
            click.echo(f"📥 Downloading clip: {clip_url}")
            clip_path = enhancer_v1.download_clip(clip_url)
        else:
            click.echo("❌ Cannot download clip - v1 enhancer not available")
            return
        
        if not clip_path:
            click.echo("❌ Failed to download clip")
            return
        
        click.echo(f"🎬 Downloaded to: {clip_path}")
        
        # Enhance with v2 system
        enhancer_v2 = ClipEnhancerV2()
        click.echo(f"🎨 Enhancing with preset system for {streamer}...")
        
        enhanced_path, telemetry = enhancer_v2.enhance_clip(clip_path, streamer_config)
        
        click.echo(f"✅ Enhanced clip ready: {enhanced_path}")
        click.echo(f"📊 Processing time: {telemetry.processing_time_ms}ms")
        click.echo(f"📊 Words rendered: {telemetry.words_rendered}")
        click.echo(f"📊 Emphasis hits: {telemetry.emphasis_hits}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
def dashboard():
    """Start the web dashboard for monitoring performance"""
    try:
        from dashboard import ClippyDashboard
        dashboard = ClippyDashboard()
        dashboard.run()
    except ImportError:
        click.echo("❌ Dashboard not available. Install with: pip install flask")
    except Exception as e:
        click.echo(f"❌ Error starting dashboard: {e}")


@cli.command()
@click.argument('video_path')
@click.option('--streamer', required=True, help='Streamer name from config')
def upload(video_path, streamer):
    """Upload a video to TikTok for a specific streamer"""
    click.echo("❌ TikTok auto-uploading has been removed. Enhanced clips are saved to folders for manual posting.")
    click.echo(f"📁 Check: clips/{streamer}/enhanced/")


@cli.command()
def list_streamers():
    """List all configured streamers and their status"""
    try:
        bot = TwitchClipBot()
        streamers = bot.config.get('streamers', [])
        
        if not streamers:
            click.echo("❌ No streamers configured")
            return
        
        click.echo("📺 Configured Streamers:")
        click.echo("=" * 50)
        
        for streamer in streamers:
            status = "🟢 ENABLED" if streamer.get('enabled', False) else "🔴 DISABLED"
            name = streamer.get('name')
            twitch_user = streamer.get('twitch_username', name)
            tiktok_user = streamer.get('tiktok_account', {}).get('username', f"{name}_clippy")
            max_posts = streamer.get('tiktok_account', {}).get('max_posts_per_day', 3)
            style = streamer.get('enhancement', {}).get('style', 'mrbeast')
            
            click.echo(f"\n{status} {name}")
            click.echo(f"   Twitch: @{twitch_user}")
            click.echo(f"   TikTok: @{tiktok_user}")
            click.echo(f"   Max Posts/Day: {max_posts}")
            click.echo(f"   Style: {style}")
            
            # Show thresholds
            thresholds = streamer.get('thresholds', {})
            click.echo(f"   Thresholds: {thresholds.get('chat_spike', 30)} msgs/sec, {thresholds.get('viewer_increase', 20)}% viewer jump")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.argument('streamer_name')
@click.option('--enable/--disable', default=True, help='Enable or disable streamer')
def toggle_streamer(streamer_name, enable):
    """Enable or disable a streamer"""
    try:
        config_path = Path("config/config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Find and update streamer
        updated = False
        for streamer in config.get('streamers', []):
            if streamer.get('name') == streamer_name:
                streamer['enabled'] = enable
                updated = True
                break
        
        if not updated:
            click.echo(f"❌ Streamer '{streamer_name}' not found")
            return
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        status = "enabled" if enable else "disabled"
        click.echo(f"✅ Streamer '{streamer_name}' {status}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option('--days', default=7, help='Number of days to show stats for')
def stats(days):
    """Show performance statistics"""
    click.echo("❌ TikTok stats not available (auto-uploading removed)")
    click.echo("📁 Enhanced clips are saved to: clips/[streamer]/enhanced/")


@cli.command()
@click.argument('streamer_name')
@click.option('--chat-spike', type=float, help='Messages per second threshold')
@click.option('--viewer-increase', type=float, help='Viewer percentage increase threshold')
@click.option('--time-window', type=int, help='Time window in seconds for chat analysis')
def set_thresholds(streamer_name, chat_spike, viewer_increase, time_window):
    """Set detection thresholds for a specific streamer"""
    try:
        config_path = Path("config/config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Find streamer
        streamer_found = False
        for streamer in config.get('streamers', []):
            if streamer.get('name') == streamer_name:
                if 'thresholds' not in streamer:
                    streamer['thresholds'] = {}
                
                # Update thresholds
                if chat_spike is not None:
                    streamer['thresholds']['chat_spike'] = chat_spike
                    click.echo(f"✅ Set chat spike threshold to {chat_spike} msgs/sec")
                
                if viewer_increase is not None:
                    streamer['thresholds']['viewer_increase'] = viewer_increase
                    click.echo(f"✅ Set viewer increase threshold to {viewer_increase}%")
                
                if time_window is not None:
                    streamer['thresholds']['time_window'] = time_window
                    click.echo(f"✅ Set time window to {time_window} seconds")
                
                streamer_found = True
                break
        
        if not streamer_found:
            click.echo(f"❌ Streamer '{streamer_name}' not found")
            return
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Show current thresholds
        updated_streamer = next(s for s in config['streamers'] if s['name'] == streamer_name)
        thresholds = updated_streamer.get('thresholds', {})
        
        click.echo(f"\n🔧 Current thresholds for {streamer_name}:")
        click.echo(f"   Chat spike: {thresholds.get('chat_spike', 'using global default')} msgs/sec")
        click.echo(f"   Viewer increase: {thresholds.get('viewer_increase', 'using global default')}%")
        click.echo(f"   Time window: {thresholds.get('time_window', 'using global default')} seconds")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option('--enable/--disable', default=True, help='Enable or disable viral algorithm')
@click.option('--score', type=float, help='Set viral score threshold (0.0-2.0+)')
def toggle_viral_algorithm(enable, score):
    """Toggle the advanced viral detection algorithm"""
    try:
        config_path = Path("config/config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update algorithm setting
        if 'global' not in config:
            config['global'] = {}
        
        config['global']['use_test_virality_alg'] = enable
        
        if score is not None:
            if 'viral_algorithm' not in config['global']:
                config['global']['viral_algorithm'] = {}
            config['global']['viral_algorithm']['score_threshold'] = score
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        status = "enabled" if enable else "disabled"
        click.echo(f"✅ Viral detection algorithm {status}")
        
        if score is not None:
            click.echo(f"✅ Viral score threshold set to {score}")
        
        # Show current algorithm settings
        alg_config = config['global'].get('viral_algorithm', {})
        click.echo(f"\n🎯 Current Settings:")
        click.echo(f"   Algorithm: {'Advanced Viral Detection' if enable else 'Legacy Thresholds'}")
        click.echo(f"   Score threshold: {alg_config.get('score_threshold', 1.0)}")
        click.echo(f"   Min unique chatters: {alg_config.get('min_unique_chatters', 30)}")
        click.echo(f"   Cooldown: {alg_config.get('cooldown_seconds', 120)}s")
        
        weights = alg_config.get('weights', {})
        click.echo(f"   Weights: Chat {weights.get('chat_velocity', 0.4):.0%}, "
                  f"Viewers {weights.get('viewer_delta', 0.3):.0%}, "
                  f"Events {weights.get('engagement_events', 0.2):.0%}, "
                  f"Follows {weights.get('follow_rate', 0.1):.0%}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
def test_viral_algorithm():
    """Test the viral detection algorithm"""
    try:
        # Import and run the test
        from viral_detector import test_viral_detector
        test_viral_detector()
        
    except ImportError:
        click.echo("❌ Viral detector not available")
    except Exception as e:
        click.echo(f"❌ Test failed: {e}")


@cli.command()
@click.option('--streamer', default='theburntpeanut', help='Streamer name')
@click.option('--days', default=7, help='Days to look back for clips')
@click.option('--max-clips', type=int, help='Maximum number of clips to include (default: ALL clips)')
@click.option('--min-clips', type=int, help='Minimum number of clips to include')
@click.option('--no-intro', is_flag=True, help='Skip intro')
@click.option('--no-transitions', is_flag=True, help='Skip transitions')
@click.option('--output', help='Custom output path')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompts')
def weekly_compilation(streamer, days, max_clips, min_clips, no_intro, no_transitions, output, yes):
    """Create a weekly compilation video from clips for YouTube"""
    try:
        from weekly_compilation import WeeklyCompilationCreator
        from moviepy.editor import VideoFileClip
        
        click.echo("=" * 60)
        click.echo("🎬 CLIPPPY WEEKLY COMPILATION CREATOR")
        click.echo("=" * 60)
        click.echo()
        
        creator = WeeklyCompilationCreator()
        
        # Preview available clips
        clips_data = creator.get_raw_clips_for_streamer(
            streamer,
            days_back=days,
            min_clips=min_clips,
            max_clips=max_clips,
            use_enhanced=True  # Use enhanced clips for now since raw clips aren't saved yet
        )
        
        if not clips_data:
            click.echo(f"❌ No clips found for {streamer}")
            click.echo(f"   Looked in: clips/{streamer}/")
            click.echo()
            click.echo("💡 Make sure enhanced clips exist.")
            click.echo("   Create some clips first with: python twitch_clip_bot.py start")
            return
        
        click.echo(f"📊 Found {len(clips_data)} clips:")
        total_duration = 0
        
        for idx, (clip_path, clip_time) in enumerate(clips_data, 1):
            try:
                clip = VideoFileClip(str(clip_path))
                duration = clip.duration
                clip.close()
                total_duration += duration
                click.echo(f"   {idx}. {clip_path.name} - {duration:.1f}s - {clip_time.strftime('%Y-%m-%d %H:%M')}")
            except:
                click.echo(f"   {idx}. {clip_path.name} - (could not read duration)")
        
        click.echo()
        click.echo(f"⏱️ Estimated compilation length: {total_duration/60:.1f} minutes")
        click.echo(f"🎬 Options: Intro={'Yes' if not no_intro else 'No'}, Transitions={'Yes' if not no_transitions else 'No'}")
        click.echo()
        
        if not yes and not click.confirm("📹 Create compilation from these clips?", default=True):
            click.echo("❌ Cancelled")
            return
        
        # Create compilation
        result = creator.create_compilation(
            streamer_name=streamer,
            days_back=days,
            output_path=output,
            add_intro=not no_intro,
            add_transitions=not no_transitions,
            max_clips=max_clips,
            min_clips=min_clips,
            use_enhanced=True  # Use enhanced clips for now
        )
        
        if result:
            click.echo()
            click.echo("=" * 60)
            click.echo(f"✅ Compilation created successfully!")
            click.echo(f"📁 Output: {result}")
            click.echo()
            
            # Ask if they want to queue for YouTube
            if yes or click.confirm("📺 Queue this for YouTube upload?", default=True):
                try:
                    from youtube_uploader import YouTubeUploader
                    uploader = YouTubeUploader()
                    uploader.queue_video(
                        video_path=result,
                        streamer_name=streamer,
                        is_compilation=True,
                        priority=10  # High priority for compilations
                    )
                    click.echo("✅ Queued for YouTube!")
                except Exception as e:
                    click.echo(f"⚠️ Failed to queue: {e}")
            
            click.echo("=" * 60)
        else:
            click.echo()
            click.echo("❌ Failed to create compilation")
            
    except ImportError as e:
        click.echo("❌ Required libraries not available")
        click.echo(f"   Error: {e}")
        click.echo("   Install with: pip install moviepy")
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


@cli.command()
def youtube_auth():
    """Authenticate with YouTube API"""
    try:
        from youtube_uploader import YouTubeUploader
        
        click.echo("=" * 60)
        click.echo("🔐 YOUTUBE AUTHENTICATION")
        click.echo("=" * 60)
        click.echo()
        
        uploader = YouTubeUploader()
        
        if uploader.authenticate():
            click.echo("✅ Authentication successful!")
            click.echo()
            click.echo("You can now upload videos to YouTube.")
        else:
            click.echo("❌ Authentication failed")
            click.echo()
            click.echo("📝 Setup instructions:")
            click.echo("   1. Go to: https://console.cloud.google.com/")
            click.echo("   2. Create a new project")
            click.echo("   3. Enable YouTube Data API v3")
            click.echo("   4. Create OAuth 2.0 credentials")
            click.echo("   5. Download credentials.json")
            click.echo("   6. Save as: config/youtube_credentials.json")
            
    except ImportError:
        click.echo("❌ YouTube API not available")
        click.echo("   Install with: pip install google-api-python-client google-auth-oauthlib")
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
def youtube_queue_status():
    """Check YouTube upload queue status"""
    try:
        from youtube_uploader import YouTubeUploader
        
        uploader = YouTubeUploader()
        status = uploader.get_queue_status()
        
        click.echo("=" * 60)
        click.echo("📊 YOUTUBE UPLOAD QUEUE STATUS")
        click.echo("=" * 60)
        click.echo()
        click.echo(f"📋 Total items: {status['total']}")
        click.echo(f"⏳ Pending: {status['pending']}")
        click.echo(f"✅ Uploaded: {status['uploaded']}")
        click.echo(f"❌ Failed: {status['failed']}")
        click.echo()
        
        if status['next_upload']:
            next_item = status['next_upload']
            scheduled_time = datetime.fromisoformat(next_item['metadata']['scheduled_time'])
            click.echo("📅 Next scheduled upload:")
            click.echo(f"   Title: {next_item['metadata']['title']}")
            click.echo(f"   Time: {scheduled_time.strftime('%A, %B %d at %I:%M %p')}")
            click.echo(f"   Video: {next_item['video_path']}")
        else:
            click.echo("📭 No pending uploads")
        
    except ImportError:
        click.echo("❌ YouTube uploader not available")
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
@click.option('--max-uploads', default=1, help='Max videos to upload in this run')
def youtube_process_queue(max_uploads):
    """Process YouTube upload queue"""
    try:
        from youtube_uploader import YouTubeUploader
        
        click.echo("=" * 60)
        click.echo("📺 PROCESSING YOUTUBE QUEUE")
        click.echo("=" * 60)
        click.echo()
        
        uploader = YouTubeUploader()
        if not uploader.authenticate():
            click.echo("❌ Authentication failed")
            return
        
        uploader.process_queue(max_uploads=max_uploads)
        
        # Show updated status
        status = uploader.get_queue_status()
        click.echo()
        click.echo(f"✅ Processing complete!")
        click.echo(f"   Pending: {status['pending']}")
        click.echo(f"   Uploaded: {status['uploaded']}")
        
    except ImportError:
        click.echo("❌ YouTube uploader not available")
    except Exception as e:
        click.echo(f"❌ Error: {e}")


if __name__ == "__main__":
    cli()
