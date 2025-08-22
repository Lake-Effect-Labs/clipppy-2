#!/usr/bin/env python3
"""
Twitch Clip Bot - Automatically creates clips when metrics spike
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import click
import requests
import websockets
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TwitchClipBot:
    def __init__(self):
        self.client_id = os.getenv('TWITCH_CLIENT_ID')
        self.client_secret = os.getenv('TWITCH_CLIENT_SECRET')
        self.broadcaster_id = os.getenv('TWITCH_BROADCASTER_ID')
        self.oauth_token = os.getenv('TWITCH_OAUTH_TOKEN')
        
        if not all([self.client_id, self.client_secret, self.broadcaster_id]):
            raise ValueError("Missing required environment variables. Check your .env file.")
        
        self.access_token = None
        self.headers = {}
        
        # Monitoring data
        self.chat_messages = deque(maxlen=1000)  # Store recent chat messages with timestamps
        self.viewer_history = deque(maxlen=60)   # Store viewer counts for last 60 checks
        self.last_clip_time = 0
        self.clip_cooldown = 60  # Minimum seconds between clips
        
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
        try:
            # Add cache-busting parameter to get fresh data
            import time
            url = f"https://api.twitch.tv/helix/streams?user_id={self.broadcaster_id}&_t={int(time.time())}"
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
        try:
            url = f"https://api.twitch.tv/helix/users?id={self.broadcaster_id}"
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
            
            url = f"https://api.twitch.tv/helix/clips?broadcaster_id={self.broadcaster_id}"
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
        
        logger.info(f"Starting monitoring for {broadcaster_name} (ID: {self.broadcaster_id})")
        
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
def start():
    """Start monitoring the broadcaster for clip opportunities"""
    try:
        bot = TwitchClipBot()
        asyncio.run(bot.start_monitoring())
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}")
        click.echo("Please check your .env file has all required variables.")
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
        click.echo(f"   Client ID: {'✅ Set' if bot.client_id else '❌ Missing'}")
        click.echo(f"   Client Secret: {'✅ Set' if bot.client_secret else '❌ Missing'}")
        click.echo(f"   Broadcaster ID: {'✅ Set' if bot.broadcaster_id else '❌ Missing'}")
        click.echo(f"   OAuth Token: {'✅ Set' if bot.oauth_token and bot.oauth_token != 'your_oauth_token_here' else '❌ Missing (needed for clips)'}")
        
        if all([bot.client_id, bot.client_secret, bot.broadcaster_id]):
            click.echo("\n🔑 Testing authentication...")
            
            async def test_auth():
                if await bot.authenticate():
                    broadcaster_name = bot.get_broadcaster_name()
                    if broadcaster_name:
                        click.echo(f"✅ Authentication successful for: {broadcaster_name}")
                        
                        if not bot.oauth_token or bot.oauth_token == 'your_oauth_token_here':
                            click.echo("\n⚠️  Note: You're using client credentials.")
                            click.echo("   For clip creation, you need a user OAuth token with 'clips:edit' scope.")
                            click.echo("   Run 'python twitch_clip_bot.py oauth-help' for instructions.")
                    else:
                        click.echo("⚠️  Authentication successful but couldn't get broadcaster name")
                else:
                    click.echo("❌ Authentication failed")
            
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


if __name__ == "__main__":
    cli()
