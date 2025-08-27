#!/usr/bin/env python3
"""
Twitch Clip Bot - Phase 1 Complete System
Automatically creates clips when metrics spike and uploads to TikTok
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from pathlib import Path

import click
import requests
import websockets
import yaml

# Import our enhanced modules
try:
    from src.clip_enhancer import ClipEnhancer, check_dependencies
    from services.tiktok_uploader import TikTokUploader
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    ClipEnhancer = None
    check_dependencies = None
    TikTokUploader = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TwitchClipBot:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize bot with YAML configuration"""
        self.config_path = Path(config_path)
        self.load_config()
        
        # Initialize components
        self.tiktok_uploader = TikTokUploader(config_path) if TikTokUploader else None
        self.clip_enhancer = ClipEnhancer() if ClipEnhancer else None
        
        # Twitch API setup from config
        twitch_config = self.config.get('twitch', {})
        self.client_id = twitch_config.get('client_id')
        self.client_secret = twitch_config.get('client_secret')
        self.oauth_token = twitch_config.get('oauth_token')
        
        if not all([self.client_id, self.client_secret]):
            raise ValueError("Missing Twitch credentials in config.yaml")
        
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
        
    async def authenticate(self) -> bool:
        """Get OAuth access token from Twitch API"""
        try:
            # Use user OAuth token if provided, otherwise fall back to client credentials
            if self.oauth_token:
                # Use provided user OAuth token
                self.access_token = self.oauth_token
                logger.info("Using provided OAuth token")
            else:
                # Fall back to client credentials (limited permissions)
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
                logger.info("Using client credentials (limited clip permissions)")
            
            self.headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Client-Id': self.client_id,
                'Content-Type': 'application/json'
            }
            
            logger.info("Successfully authenticated with Twitch API")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def get_stream_data(self) -> Optional[Dict]:
        """Get current stream data including viewer count"""
        if not self.current_streamer:
            logger.error("No current streamer set")
            return None
            
        try:
            # Add cache-busting parameter to get fresh data
            import time
            broadcaster_id = self.current_streamer.get('broadcaster_id')
            url = f"https://api.twitch.tv/helix/streams?user_id={broadcaster_id}&_t={int(time.time())}"
            response = requests.get(url, headers=self.headers)
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
                
        except requests.RequestException as e:
            logger.error(f"Failed to get stream data: {e}")
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
                
                logger.info(f"Connected to chat for #{broadcaster_name}")
                
                async for message in websocket:
                    if message.startswith("PING"):
                        await websocket.send("PONG :tmi.twitch.tv")
                        continue
                    
                    if "PRIVMSG" in message:
                        # Record chat message timestamp
                        current_time = time.time()
                        self.chat_messages.append(current_time)
                        
                        # Parse username and message for logging
                        try:
                            parts = message.split(":", 2)
                            if len(parts) >= 3:
                                user_info = parts[1].split("!")[0]
                                chat_message = parts[2].strip()
                                logger.debug(f"Chat: {user_info}: {chat_message}")
                        except:
                            pass
                            
        except Exception as e:
            logger.error(f"Chat monitoring error: {e}")
    
    def get_broadcaster_name(self) -> Optional[str]:
        """Get broadcaster username from user ID"""
        if not self.current_streamer:
            return None
            
        try:
            broadcaster_id = self.current_streamer.get('broadcaster_id')
            url = f"https://api.twitch.tv/helix/users?id={broadcaster_id}"
            response = requests.get(url, headers=self.headers)
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
        # Check chat activity spike
        chat_mps = self.calculate_chat_mps(10)
        if chat_mps > 30:
            return True, f"Chat spike detected: {chat_mps:.1f} messages/second"
        
        # Check viewer count spike
        viewer_change, current_viewers = self.calculate_viewer_change()
        if viewer_change > 20:
            return True, f"Viewer spike detected: +{viewer_change:.1f}% ({current_viewers} viewers)"
        
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
            response = requests.post(url, headers=self.headers)
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
                
                # Auto-enhance clip if enabled
                if self.auto_enhance and self.enhancer:
                    logger.info("🎨 Auto-enhancing clip...")
                    use_captions = os.getenv('USE_CAPTIONS', 'true').lower() == 'true'
                    enhanced_result = self.enhancer.auto_enhance_clip(clip_url, reason, use_captions=use_captions)
                    if enhanced_result:
                        logger.info(f"✨ Enhanced clip saved: {enhanced_result['enhanced_path']}")
                        if enhanced_result.get('has_captions'):
                            logger.info("   Features: Animated captions with speech-to-text")
                        if enhanced_result.get('text_overlay'):
                            logger.info(f"   Text: {enhanced_result['text_overlay']}")
                        if enhanced_result.get('sound_effect'):
                            logger.info(f"   Sound: {enhanced_result['sound_effect']}")
                
                return clip_url
            
        except requests.RequestException as e:
            if e.response and e.response.status_code == 429:
                logger.warning("Rate limited by Twitch API")
            elif e.response and e.response.status_code == 401:
                logger.error("❌ Unauthorized: Need user OAuth token with 'clips:edit' scope for clip creation")
                logger.error("   Client credentials cannot create clips. You need to get a user OAuth token.")
            else:
                logger.error(f"Failed to create clip: {e}")
        
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
        
        # Start chat monitoring in background
        chat_task = asyncio.create_task(self.monitor_chat(broadcaster_name))
        
        try:
            while True:
                # Get stream data
                stream_data = self.get_stream_data()
                
                if stream_data is None:
                    logger.info("Stream is offline, waiting...")
                    await asyncio.sleep(30)
                    continue
                
                # Record viewer count
                viewer_count = stream_data['viewer_count']
                self.viewer_history.append(viewer_count)
                
                # Check for spikes
                should_clip, reason = self.detect_spikes()
                
                if should_clip:
                    logger.info(f"🚨 Spike detected! {reason}")
                    self.create_clip(reason)
                
                # Log current stats
                chat_mps = self.calculate_chat_mps(10)
                viewer_change, _ = self.calculate_viewer_change()
                logger.info(f"📊 Viewers: {viewer_count}, Chat: {chat_mps:.1f} MPS, Change: {viewer_change:+.1f}%")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            chat_task.cancel()


# CLI Interface
@click.group()
def cli():
    """Twitch Clip Bot - Automatically create clips when metrics spike"""
    pass


@cli.command()
@click.option('--streamer', help='Specific streamer to monitor (default: all enabled)')
def start(streamer):
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
        click.echo(f"❌ Configuration error: {e}")
        click.echo("Please check your config.yaml file.")
    except Exception as e:
        click.echo(f"❌ Error: {e}")


@cli.command()
def testclip():
    """Create a test clip manually"""
    try:
        bot = TwitchClipBot()
        
        async def test():
            if await bot.authenticate():
                clip_url = bot.create_clip("Manual test clip")
                if clip_url:
                    click.echo(f"✅ Test clip created: {clip_url}")
                else:
                    click.echo("❌ Failed to create test clip")
            else:
                click.echo("❌ Authentication failed")
        
        asyncio.run(test())
        
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}")
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
        click.echo(f"   Twitch OAuth Token: {'✅ Set' if bot.oauth_token and bot.oauth_token != 'your_twitch_oauth_token' else '❌ Missing (needed for clips)'}")
        
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
        
        # Show TikTok status
        click.echo(f"\n📱 TikTok Integration: {'✅ Ready' if bot.tiktok_uploader else '❌ Not available'}")
        
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
                    
                    if not bot.oauth_token or bot.oauth_token == 'your_twitch_oauth_token':
                        click.echo("\n⚠️  Note: You're using client credentials.")
                        click.echo("   For clip creation, you need a user OAuth token.")
                        click.echo("   Run: python twitch_clip_bot.py oauth-help --generate-url")
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
            click.echo("1. Run: python twitch_clip_bot.py oauth-help --generate-url")
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
def dashboard():
    """Start the web dashboard for monitoring performance"""
    try:
        from services.dashboard import ClippyDashboard
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
    try:
        bot = TwitchClipBot()
        
        if not bot.tiktok_uploader:
            click.echo("❌ TikTok uploader not available")
            return
        
        streamer_config = bot.get_streamer_config(streamer)
        if not streamer_config:
            click.echo(f"❌ Streamer '{streamer}' not found in config")
            return
        
        success = bot.tiktok_uploader.upload_to_tiktok(video_path, streamer)
        
        if success:
            click.echo(f"✅ Video queued for upload to @{streamer_config.get('tiktok_account', {}).get('username', f'{streamer}_clippy')}")
        else:
            click.echo("❌ Upload failed")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}")


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
    try:
        bot = TwitchClipBot()
        
        if bot.tiktok_uploader:
            enabled_streamers = bot.get_enabled_streamers()
            
            click.echo(f"📊 Performance Stats (Last {days} days)")
            click.echo("=" * 50)
            
            total_uploads = 0
            for streamer in enabled_streamers:
                stats = bot.tiktok_uploader.get_account_stats(streamer['name'])
                total_uploads += stats['total_uploads']
                
                click.echo(f"\n🎬 {streamer['name']}")
                click.echo(f"   Total uploads: {stats['total_uploads']}")
                click.echo(f"   Uploads today: {stats['uploads_today']}")
                
                for account in stats['accounts']:
                    click.echo(f"   Account uploads today: {account['uploads_today']}")
            
            click.echo(f"\n🎯 Total Network Uploads: {total_uploads}")
            
        else:
            click.echo("❌ Statistics not available - TikTok uploader not initialized")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if os.getenv('OPENAI_API_KEY'):
            click.echo("   pip install openai  # Optional for AI text generation")
    
    # Check OpenAI API key
    if os.getenv('OPENAI_API_KEY'):
        click.echo("✅ OpenAI API key configured")
    else:
        click.echo("⚠️  No OpenAI API key found (optional)")
        click.echo("   Add OPENAI_API_KEY=your_key to .env for AI text generation")
    
    # Test enhancement
    click.echo("\n🧪 Testing enhancement capabilities...")
    try:
        enhancer = ClipEnhancer()
        test_text = enhancer.generate_viral_text("test")
        click.echo(f"✅ Text generation: {test_text}")
        
        click.echo("✅ Clip enhancement ready!")
        click.echo("\n📚 Usage:")
        click.echo("   python twitch_clip_bot.py enhance <clip_url>")
        click.echo("   python twitch_clip_bot.py enhance <clip_url> --text 'CUSTOM TEXT' --sound airhorn")
        
    except Exception as e:
        click.echo(f"❌ Enhancement test failed: {e}")


if __name__ == "__main__":
    cli()
