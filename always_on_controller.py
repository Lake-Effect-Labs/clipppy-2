#!/usr/bin/env python3
"""
🚀 Always-On Multi-Streamer Controller
Monitors all configured streamers and spawns individual listeners when they go live.
Manages concurrent enhancement and centralized TikTok posting.
"""

import asyncio
import logging
import subprocess
import time
import yaml
import os
import requests
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import queue
import json

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load environment variables at startup
load_env_file()

# Configure logging with proper UTF-8 encoding for Windows emoji support
import sys
import io

# Ensure UTF-8 encoding for console output on Windows
if sys.platform == 'win32':
    # Reconfigure stdout to handle UTF-8 properly
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Create logs directory
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Configure logging with UTF-8 support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/always_on_controller.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class StreamerStatus:
    name: str
    broadcaster_id: str
    is_live: bool
    last_checked: datetime
    listener_process: subprocess.Popen = None
    game_name: str = "Unknown"
    viewer_count: int = 0
    twitch_handle: str = ""

class AlwaysOnController:
    """Master controller for always-on multi-streamer monitoring"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.load_config()
        
        # Twitch API setup
        self.client_id = os.getenv('TWITCH_CLIENT_ID')
        self.client_secret = os.getenv('TWITCH_CLIENT_SECRET')
        self.access_token = None
        self.headers = {}
        
        # State tracking
        self.streamer_status: Dict[str, StreamerStatus] = {}
        self.running = False
        self.check_interval = 1800  # 30 minutes
        
        # Enhancement queue management
        self.enhancement_queue = queue.Queue()
        self.tiktok_posting_queue = queue.Queue()
        self.enhancement_workers = []
        self.max_concurrent_enhancements = 3
        
        # Initialize streamers
        self.initialize_streamers()
        
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("✅ Configuration loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            raise
    
    def initialize_streamers(self):
        """Initialize streamer status tracking"""
        for streamer in self.config.get('streamers', []):
            if streamer.get('enabled', False):
                status = StreamerStatus(
                    name=streamer['name'],
                    broadcaster_id=streamer['broadcaster_id'],
                    is_live=False,
                    last_checked=datetime.now() - timedelta(hours=1),
                    twitch_handle=streamer.get('twitch_username', streamer['name'])
                )
                self.streamer_status[streamer['name']] = status
                logger.info(f"📋 Registered streamer: {streamer['name']} ({streamer['broadcaster_id']})")
    
    async def authenticate_twitch(self) -> bool:
        """Authenticate with Twitch API"""
        try:
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
            self.headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Client-Id': self.client_id,
                'Content-Type': 'application/json'
            }
            
            logger.info("🔑 Successfully authenticated with Twitch API")
            return True
            
        except Exception as e:
            logger.error(f"❌ Twitch authentication failed: {e}")
            return False
    
    async def check_streamer_status(self, streamer_name: str) -> bool:
        """Check if a specific streamer is live"""
        try:
            status = self.streamer_status[streamer_name]
            
            # Get stream data
            url = f"https://api.twitch.tv/helix/streams?user_id={status.broadcaster_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            is_live = len(data.get('data', [])) > 0
            
            if is_live:
                stream_data = data['data'][0]
                status.game_name = stream_data.get('game_name', 'Unknown')
                status.viewer_count = stream_data.get('viewer_count', 0)
                
                # Special handling for "I'm Only Sleeping" - treat as offline
                if status.game_name == "I'm Only Sleeping":
                    logger.info(f"😴 {streamer_name} is in 'I'm Only Sleeping' mode - treating as offline")
                    status.last_checked = datetime.now()
                    return False
                
                logger.info(f"🔴 {streamer_name} is LIVE - {status.game_name} ({status.viewer_count:,} viewers)")
            else:
                logger.info(f"⚫ {streamer_name} is offline")
            
            # Don't update status.is_live here - let the caller handle state changes
            status.last_checked = datetime.now()
            
            return is_live
            
        except Exception as e:
            logger.error(f"❌ Failed to check {streamer_name} status: {e}")
            # If we can't check status due to network error, assume offline
            if "Failed to resolve" in str(e) or "Max retries exceeded" in str(e):
                logger.warning(f"🌐 Network error checking {streamer_name} - assuming offline")
                return False
            return False
    
    def spawn_listener(self, streamer_name: str):
        """Spawn a listener process for a live streamer in a new PowerShell window"""
        try:
            status = self.streamer_status[streamer_name]
            
            # Check if listener is already running by searching for the process
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'python' in proc.info['name'].lower() and proc.info['cmdline']:
                        cmdline = ' '.join(proc.info['cmdline'])
                        if 'twitch_clip_bot.py' in cmdline and f'--streamer {streamer_name}' in cmdline:
                            logger.info(f"✅ Listener for {streamer_name} is already running (PID: {proc.info['pid']})")
                            return
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Create PowerShell command to open new window with the listener
            working_dir = os.getcwd()
            window_title = f"Clipppy - {streamer_name.upper()}"
            ps_cmd = f"Set-Location '{working_dir}'; python twitch_clip_bot.py start --streamer {streamer_name} --always-on-mode"
            
            # Use Start-Process to create a new visible PowerShell window with custom title
            cmd = [
                'powershell', '-Command',
                f'Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd \'{working_dir}\'; $Host.UI.RawUI.WindowTitle = \'{window_title}\'; python twitch_clip_bot.py start --streamer {streamer_name} --always-on-mode"'
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
                creationflags=subprocess.CREATE_NEW_CONSOLE if hasattr(subprocess, 'CREATE_NEW_CONSOLE') else 0
            )
            
            # Give it a moment to start
            import time
            time.sleep(2)
            
            status.listener_process = process
            logger.info(f"🚀 Spawned visible PowerShell listener for {streamer_name} (PID: {process.pid})")
            
        except Exception as e:
            logger.error(f"❌ Failed to spawn listener for {streamer_name}: {e}")
    
    def kill_listener(self, streamer_name: str):
        """Kill the listener process for a streamer and close PowerShell windows"""
        try:
            status = self.streamer_status[streamer_name]
            if status.listener_process:
                # First try to gracefully terminate
                status.listener_process.terminate()
                try:
                    status.listener_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    status.listener_process.kill()
                
                # Kill any python processes and PowerShell windows for this streamer
                try:
                    window_title = f"Clipppy - {streamer_name.upper()}"
                    
                    # Method 1: Kill PowerShell windows by title
                    subprocess.run([
                        'taskkill', '/F', '/FI', 
                        f'WINDOWTITLE eq {window_title}'
                    ], capture_output=True, timeout=5)
                    
                    # Method 2: Use taskkill to kill python processes with specific command line
                    subprocess.run([
                        'taskkill', '/F', '/FI', 
                        f'IMAGENAME eq python.exe', '/FI',
                        f'COMMANDLINE eq *{streamer_name}*'
                    ], capture_output=True, timeout=5)
                    
                    # Method 3: Use wmic as backup for any remaining processes
                    subprocess.run([
                        'wmic', 'process', 'where', 
                        f'(name="python.exe" or name="powershell.exe") and commandline like "%{streamer_name}%"',
                        'delete'
                    ], capture_output=True, timeout=5)
                    
                    logger.info(f"✅ Cleaned up all processes for {streamer_name}")
                    
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Cleanup warning for {streamer_name}: {cleanup_error}")
                
                logger.info(f"💀 Killed listener for {streamer_name}")
                status.listener_process = None
                
        except Exception as e:
            logger.error(f"❌ Failed to kill listener for {streamer_name}: {e}")
    
    def start_enhancement_workers(self):
        """Start background enhancement worker threads"""
        for i in range(self.max_concurrent_enhancements):
            worker = threading.Thread(
                target=self._enhancement_worker,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.enhancement_workers.append(worker)
            logger.info(f"🎨 Started enhancement worker {i}")
    
    def _enhancement_worker(self, worker_id: int):
        """Background enhancement worker"""
        logger.info(f"🎨 Enhancement worker {worker_id} started")
        
        while self.running:
            try:
                # Get enhancement job from queue
                job = self.enhancement_queue.get(timeout=5)
                if job is None:  # Shutdown signal
                    break
                
                logger.info(f"🎬 Worker {worker_id} processing: {job['clip_url']}")
                
                # Process enhancement (import here to avoid circular imports)
                from clip_enhancer_v2 import ClipEnhancerV2 as ClipEnhancer
                from clip_enhancer_v2 import ClipEnhancerV2
                
                enhancer_v1 = ClipEnhancer()
                enhancer_v2 = ClipEnhancerV2()
                
                # Download clip
                clip_path = enhancer_v1.download_clip(job['clip_url'])
                if not clip_path:
                    logger.error(f"❌ Worker {worker_id} failed to download clip")
                    continue
                
                # Enhance clip
                enhanced_path, telemetry = enhancer_v2.enhance_clip(
                    clip_path, job['streamer_config']
                )
                
                if enhanced_path:
                    # Add to TikTok posting queue
                    posting_job = {
                        'video_path': enhanced_path,
                        'streamer_name': job['streamer_name'],
                        'streamer_handle': job['streamer_handle'],
                        'game_name': job['game_name'],
                        'clip_url': job['clip_url']
                    }
                    self.tiktok_posting_queue.put(posting_job)
                    logger.info(f"✅ Worker {worker_id} completed enhancement: {enhanced_path}")
                else:
                    logger.error(f"❌ Worker {worker_id} failed enhancement")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Enhancement worker {worker_id} error: {e}")
    
    def start_tiktok_poster(self):
        """Start TikTok posting worker"""
        worker = threading.Thread(target=self._tiktok_poster, daemon=True)
        worker.start()
        logger.info("📱 Started TikTok posting worker")
    
    def start_queue_monitor(self):
        """Start file-based queue monitor for listener communications"""
        monitor = threading.Thread(target=self._queue_monitor, daemon=True)
        monitor.start()
        logger.info("📂 Started enhancement queue monitor")
    
    def _tiktok_poster(self):
        """Background TikTok posting worker"""
        logger.info("📱 TikTok posting worker started")
        
        while self.running:
            try:
                # Get posting job from queue
                job = self.tiktok_posting_queue.get(timeout=10)
                if job is None:  # Shutdown signal
                    break
                
                # Generate caption with streamer info
                caption = self.generate_tiktok_caption(
                    job['streamer_name'],
                    job['streamer_handle'], 
                    job['game_name']
                )
                
                logger.info(f"📱 Posting to TikTok: {job['streamer_name']} clip")
                logger.info(f"📝 Caption: {caption[:50]}...")
                
                # TODO: Implement actual TikTok posting
                # For now, just log the action
                logger.info(f"✅ TikTok post queued: {job['video_path']}")
                
                # Save posting record
                self.save_posting_record(job, caption)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ TikTok posting error: {e}")
    
    def _queue_monitor(self):
        """Monitor file-based enhancement queue from listeners"""
        import os
        import json
        import time
        
        logger.info("📂 Enhancement queue monitor started")
        queue_dir = "data/enhancement_queue"
        os.makedirs(queue_dir, exist_ok=True)
        
        while self.running:
            try:
                # Check for new queue files
                queue_files = [f for f in os.listdir(queue_dir) if f.endswith('.json')]
                
                for queue_file in queue_files:
                    file_path = os.path.join(queue_dir, queue_file)
                    
                    try:
                        # Read and parse the queue file
                        with open(file_path, 'r') as f:
                            job_data = json.load(f)
                        
                        # Add to enhancement queue
                        self.enhancement_queue.put(job_data)
                        
                        # Remove processed file
                        os.remove(file_path)
                        
                        logger.info(f"📂 Queued clip from {job_data['streamer_name']}: {job_data['clip_url']}")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to process queue file {queue_file}: {e}")
                        # Move problematic file to avoid infinite loop
                        try:
                            os.rename(file_path, file_path + '.error')
                        except:
                            pass
                
                # Sleep before next check
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Queue monitor error: {e}")
                time.sleep(5)
        
        logger.info("📂 Enhancement queue monitor stopped")
    
    def generate_tiktok_caption(self, streamer_name: str, streamer_handle: str, game_name: str) -> str:
        """Generate TikTok caption with streamer info and game hashtags"""
        
        # Viral phrases for different games
        game_phrases = {
            'Call of Duty': ['INSANE COD CLIP! 🔥', 'COD MOMENT THAT BROKE THE INTERNET! 💀'],
            'Fortnite': ['FORTNITE MADNESS! 🎮', 'THIS FORTNITE PLAY IS UNREAL! ⚡'],
            'Valorant': ['VALORANT CLUTCH INCOMING! 🎯', 'VALORANT PROS HATE THIS TRICK! 😱'],
            'Apex Legends': ['APEX LEGENDS CHAOS! 🚀', 'THIRD PARTY INCOMING! 💥'],
            'Rainbow Six Siege': ['R6 SIEGE MOMENT! 🎯', 'SIEGE PLAYERS UNDERSTAND! 💀'],
            'League of Legends': ['LOL MOMENT! ⚡', 'LEAGUE LEGENDS PLAY! 🔥'],
            'Minecraft': ['MINECRAFT MADNESS! ⛏️', 'BLOCK GAME GOES HARD! 🧱'],
            'Counter-Strike': ['CS CLUTCH! 💀', 'COUNTER-STRIKE CHAOS! 🔫']
        }
        
        # Get game-specific phrase or generic
        phrases = game_phrases.get(game_name, ['GAMING MOMENT! 🎮', 'STREAMER GOES CRAZY! 🔥'])
        import random
        phrase = random.choice(phrases)
        
        # Game-specific hashtags
        game_hashtags = {
            'Call of Duty': ['#cod', '#callofduty', '#warzone'],
            'Fortnite': ['#fortnite', '#fortniteclips', '#epicgames'],
            'Valorant': ['#valorant', '#valorantclips', '#riot'],
            'Apex Legends': ['#apex', '#apexlegends', '#respawn'],
            'Rainbow Six Siege': ['#r6', '#rainbowsix', '#siege'],
            'League of Legends': ['#lol', '#leagueoflegends', '#riot'],
            'Minecraft': ['#minecraft', '#minecraftclips', '#mojang'],
            'Counter-Strike': ['#cs2', '#counterstrike', '#valve']
        }
        
        # Base hashtags
        hashtags = ['#gaming', '#twitch', '#viral', '#fyp', '#clips']
        
        # Add game-specific hashtags
        if game_name in game_hashtags:
            hashtags.extend(game_hashtags[game_name])
        
        # Add streamer hashtag
        hashtags.append(f'#{streamer_name.lower()}')
        
        # Construct caption
        caption = f"{phrase}\n\nCredit: @{streamer_handle}\n\n{' '.join(hashtags)}"
        
        # Ensure under TikTok limit (150 characters)
        if len(caption) > 150:
            caption = caption[:147] + "..."
        
        return caption
    
    def save_posting_record(self, job: Dict, caption: str):
        """Save posting record for tracking"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'streamer': job['streamer_name'],
            'game': job['game_name'],
            'video_path': str(job['video_path']),
            'caption': caption,
            'clip_url': job['clip_url']
        }
        
        # Append to posting log
        log_file = Path('data/tiktok_posts.json')
        log_file.parent.mkdir(exist_ok=True)
        
        posts = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    posts = json.load(f)
            except:
                posts = []
        
        posts.append(record)
        
        with open(log_file, 'w') as f:
            json.dump(posts[-1000:], f, indent=2)  # Keep last 1000 posts
    
    async def run_status_check_cycle(self):
        """Run one complete status check cycle"""
        logger.info("🔍 Starting status check cycle...")
        
        for streamer_name, status in self.streamer_status.items():
            # Store previous state before checking
            previous_live = status.is_live
            
            # Check current status
            current_live = await self.check_streamer_status(streamer_name)
            
            # Handle state changes
            if current_live and not previous_live:
                # Streamer went live - spawn listener
                logger.info(f"🟢 {streamer_name} went LIVE! Spawning listener...")
                self.spawn_listener(streamer_name)
                
            elif not current_live and previous_live:
                # Streamer went offline - kill listener
                logger.info(f"🔴 {streamer_name} went OFFLINE! Killing listener...")
                self.kill_listener(streamer_name)
            
            elif current_live and previous_live:
                # Still live - but don't respawn unless we're sure the listener is dead
                # The spawn_listener method has its own duplicate checking via process search
                pass
            
            # Update the status after handling state changes
            status.is_live = current_live
            
            # Small delay between checks
            await asyncio.sleep(1)
        
        logger.info("✅ Status check cycle complete")
    
    async def run(self):
        """Main always-on controller loop"""
        logger.info("🚀 Starting Always-On Multi-Streamer Controller")
        
        # Authenticate with Twitch
        if not await self.authenticate_twitch():
            logger.error("❌ Failed to authenticate with Twitch")
            return
        
        # Start background workers
        self.running = True
        self.start_enhancement_workers()
        self.start_tiktok_poster()
        self.start_queue_monitor()
        
        # Main monitoring loop
        try:
            while self.running:
                # Run status check cycle
                await self.run_status_check_cycle()
                
                # Show current status
                live_count = sum(1 for s in self.streamer_status.values() if s.is_live)
                logger.info(f"📊 Status: {live_count}/{len(self.streamer_status)} streamers live")
                
                # Wait for next check
                logger.info(f"⏰ Next check in {self.check_interval} seconds...")
                
                # Use shorter sleep intervals to make shutdown more responsive
                for i in range(self.check_interval):
                    if not self.running:
                        break
                    await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested")
        except asyncio.CancelledError:
            logger.info("🛑 Shutdown requested")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Clean shutdown with proper process termination"""
        logger.info("🛑 Shutting down Always-On Controller...")
        self.running = False
        
        # Kill all listener processes and close their PowerShell windows
        for streamer_name in self.streamer_status:
            self.kill_listener(streamer_name)
        
        # Give processes time to clean up
        import time
        time.sleep(2)
        
        # Force kill any remaining PowerShell processes for our project
        try:
            import subprocess
            # Find and kill PowerShell processes running our bot
            result = subprocess.run([
                'powershell', '-Command',
                "Get-Process powershell | Where-Object {$_.CommandLine -like '*twitch_clip_bot.py*'} | Stop-Process -Force"
            ], capture_output=True, text=True, timeout=10)
            logger.info("💀 Cleaned up PowerShell listener processes")
        except Exception as e:
            logger.warning(f"⚠️ Could not force clean PowerShell processes: {e}")
        
        # Stop background workers
        logger.info("🛑 Stopping enhancement workers...")
        for _ in self.enhancement_workers:
            self.enhancement_queue.put(None)
        
        logger.info("🛑 Stopping TikTok poster...")
        self.tiktok_posting_queue.put(None)
        
        logger.info("✅ Shutdown complete")

def main():
    """Main entry point"""
    try:
        controller = AlwaysOnController()
        asyncio.run(controller.run())
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()
