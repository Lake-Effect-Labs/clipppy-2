#!/usr/bin/env python3

"""
TikTok Uploader Service
======================

Handles uploading clips to TikTok accounts with proper authentication,
rate limiting, and posting rules enforcement.
"""

import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import requests
import yaml

logger = logging.getLogger(__name__)

class TikTokUploader:
    """Manages TikTok uploads with rate limiting and account management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize TikTok uploader with configuration"""
        self.config_path = Path(config_path)
        self.load_config()
        self.upload_history = self.load_upload_history()
        
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info("✅ Configuration loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            self.config = {}
    
    def load_upload_history(self) -> Dict:
        """Load upload history from JSON file"""
        history_file = Path("data/upload_history.json")
        history_file.parent.mkdir(exist_ok=True)
        
        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.warning(f"Could not load upload history: {e}")
            return {}
    
    def save_upload_history(self):
        """Save upload history to JSON file"""
        history_file = Path("data/upload_history.json")
        try:
            with open(history_file, 'w') as f:
                json.dump(self.upload_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save upload history: {e}")
    
    def can_upload(self, streamer_name: str, tiktok_username: str) -> bool:
        """Check if we can upload for this account based on rate limits"""
        today = datetime.now().date().isoformat()
        account_key = f"{streamer_name}_{tiktok_username}"
        
        # Get today's upload count
        if account_key not in self.upload_history:
            self.upload_history[account_key] = {}
        
        today_uploads = self.upload_history[account_key].get(today, 0)
        
        # Get max posts per day from config
        max_posts = 3  # default
        for streamer in self.config.get('streamers', []):
            if streamer['name'] == streamer_name:
                max_posts = streamer.get('tiktok_account', {}).get('max_posts_per_day', 3)
                break
        
        can_post = today_uploads < max_posts
        logger.info(f"📊 {account_key}: {today_uploads}/{max_posts} posts today - {'✅ Can post' if can_post else '❌ Rate limited'}")
        
        return can_post
    
    def is_posting_hours(self, streamer_name: str) -> bool:
        """Check if current time is within posting hours for streamer"""
        current_hour = datetime.now().hour
        
        # Default posting hours (2 PM - 10 PM)
        start_hour = 14
        end_hour = 22
        
        # Get specific hours from config
        for streamer in self.config.get('streamers', []):
            if streamer['name'] == streamer_name:
                posting_hours = streamer.get('tiktok_account', {}).get('posting_hours', {})
                start_hour = posting_hours.get('start', 14)
                end_hour = posting_hours.get('end', 22)
                break
        
        is_valid_time = start_hour <= current_hour <= end_hour
        logger.info(f"⏰ Current hour: {current_hour}, Posting hours: {start_hour}-{end_hour} - {'✅ Valid' if is_valid_time else '❌ Outside hours'}")
        
        return is_valid_time
    
    def generate_caption(self, streamer_name: str, clip_title: str = None) -> str:
        """Generate TikTok caption with hashtags"""
        # Get hashtags from config
        hashtags = ["#gaming", "#viral", "#fyp"]  # defaults
        
        for streamer in self.config.get('streamers', []):
            if streamer['name'] == streamer_name:
                hashtags = streamer.get('tiktok_account', {}).get('hashtags', hashtags)
                break
        
        # Generate caption
        if clip_title:
            caption = f"{clip_title}\n\n"
        else:
            caption = f"🔥 INSANE {streamer_name.upper()} MOMENT! 🔥\n\n"
        
        caption += " ".join(hashtags)
        
        # Ensure caption is under TikTok's limit (150 characters)
        if len(caption) > 150:
            caption = caption[:147] + "..."
        
        return caption
    
    def upload_to_tiktok(self, video_path: str, streamer_name: str, clip_title: str = None) -> bool:
        """
        Upload video to TikTok (placeholder implementation)
        
        Note: This is a placeholder for the actual TikTok API integration.
        Real implementation would use TikTok's Content Posting API.
        """
        video_file = Path(video_path)
        if not video_file.exists():
            logger.error(f"❌ Video file not found: {video_path}")
            return False
        
        # Get TikTok account info
        tiktok_username = None
        for streamer in self.config.get('streamers', []):
            if streamer['name'] == streamer_name:
                tiktok_username = streamer.get('tiktok_account', {}).get('username')
                break
        
        if not tiktok_username:
            logger.error(f"❌ No TikTok account configured for streamer: {streamer_name}")
            return False
        
        # Check rate limits and posting hours
        if not self.can_upload(streamer_name, tiktok_username):
            logger.warning(f"⚠️ Rate limit reached for {tiktok_username}")
            return False
        
        if not self.is_posting_hours(streamer_name):
            logger.warning(f"⚠️ Outside posting hours for {streamer_name}")
            return False
        
        # Generate caption
        caption = self.generate_caption(streamer_name, clip_title)
        
        logger.info(f"🚀 Uploading to TikTok...")
        logger.info(f"   Account: {tiktok_username}")
        logger.info(f"   Video: {video_file.name}")
        logger.info(f"   Caption: {caption}")
        logger.info(f"   Size: {video_file.stat().st_size:,} bytes")
        
        # TODO: Implement actual TikTok API upload
        # For Phase 1, we'll simulate the upload and save the video for manual posting
        success = self.simulate_upload(video_file, tiktok_username, caption)
        
        if success:
            # Record the upload
            self.record_upload(streamer_name, tiktok_username, video_path, caption)
            logger.info(f"✅ Upload completed successfully!")
            return True
        else:
            logger.error(f"❌ Upload failed!")
            return False
    
    def simulate_upload(self, video_file: Path, tiktok_username: str, caption: str) -> bool:
        """
        Simulate TikTok upload by organizing files for manual posting
        
        In Phase 1, we'll save videos in organized folders for manual posting
        until we implement the actual TikTok API integration.
        """
        try:
            # Create upload queue directory
            upload_dir = Path(f"uploads/{tiktok_username}")
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{timestamp}_{video_file.name}"
            destination = upload_dir / new_filename
            
            # Copy video to upload queue
            import shutil
            shutil.copy2(video_file, destination)
            
            # Create caption file
            caption_file = upload_dir / f"{timestamp}_caption.txt"
            with open(caption_file, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            # Create upload instructions
            instructions_file = upload_dir / "README.txt"
            with open(instructions_file, 'w') as f:
                f.write(f"TikTok Upload Queue for @{tiktok_username}\n")
                f.write("=" * 50 + "\n\n")
                f.write("Instructions:\n")
                f.write("1. Log into TikTok account @" + tiktok_username + "\n")
                f.write("2. Upload the video files in chronological order\n")
                f.write("3. Use the corresponding caption files for each video\n")
                f.write("4. Delete files after successful upload\n\n")
                f.write("Latest upload:\n")
                f.write(f"Video: {new_filename}\n")
                f.write(f"Caption: {caption}\n")
                f.write(f"Time: {datetime.now()}\n")
            
            logger.info(f"📁 Video queued for manual upload: {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue upload: {e}")
            return False
    
    def record_upload(self, streamer_name: str, tiktok_username: str, video_path: str, caption: str):
        """Record upload in history for rate limiting"""
        today = datetime.now().date().isoformat()
        account_key = f"{streamer_name}_{tiktok_username}"
        
        # Initialize account history if needed
        if account_key not in self.upload_history:
            self.upload_history[account_key] = {}
        
        # Increment today's count
        self.upload_history[account_key][today] = self.upload_history[account_key].get(today, 0) + 1
        
        # Store upload details
        upload_details = {
            'timestamp': datetime.now().isoformat(),
            'video_path': str(video_path),
            'caption': caption,
            'account': tiktok_username
        }
        
        # Store in detailed history
        detail_key = f"{account_key}_details"
        if detail_key not in self.upload_history:
            self.upload_history[detail_key] = []
        
        self.upload_history[detail_key].append(upload_details)
        
        # Keep only last 30 days of detailed history
        cutoff_date = datetime.now() - timedelta(days=30)
        self.upload_history[detail_key] = [
            upload for upload in self.upload_history[detail_key]
            if datetime.fromisoformat(upload['timestamp']) > cutoff_date
        ]
        
        # Save updated history
        self.save_upload_history()
    
    def get_account_stats(self, streamer_name: str) -> Dict:
        """Get upload statistics for a streamer's TikTok account"""
        account_key = f"{streamer_name}_*"
        
        # Find all accounts for this streamer
        matching_accounts = [key for key in self.upload_history.keys() if key.startswith(f"{streamer_name}_") and not key.endswith("_details")]
        
        stats = {
            'total_uploads': 0,
            'uploads_today': 0,
            'uploads_this_week': 0,
            'uploads_this_month': 0,
            'accounts': []
        }
        
        today = datetime.now().date()
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        for account_key in matching_accounts:
            account_uploads = self.upload_history.get(account_key, {})
            
            account_stats = {
                'account_key': account_key,
                'uploads_today': account_uploads.get(today.isoformat(), 0),
                'total_uploads': sum(account_uploads.values())
            }
            
            stats['accounts'].append(account_stats)
            stats['uploads_today'] += account_stats['uploads_today']
            stats['total_uploads'] += account_stats['total_uploads']
        
        return stats


def test_tiktok_uploader():
    """Test the TikTok uploader functionality"""
    print("🧪 Testing TikTok Uploader")
    print("=" * 50)
    
    uploader = TikTokUploader()
    
    # Test rate limiting
    print("\n📊 Testing rate limiting...")
    can_upload = uploader.can_upload("jynxzi", "jynxzi_clippy")
    print(f"Can upload: {can_upload}")
    
    # Test posting hours
    print("\n⏰ Testing posting hours...")
    valid_time = uploader.is_posting_hours("jynxzi")
    print(f"Valid posting time: {valid_time}")
    
    # Test caption generation
    print("\n📝 Testing caption generation...")
    caption = uploader.generate_caption("jynxzi", "INSANE ACE CLUTCH!")
    print(f"Generated caption: {caption}")
    
    # Test stats
    print("\n📈 Testing account stats...")
    stats = uploader.get_account_stats("jynxzi")
    print(f"Account stats: {stats}")


if __name__ == "__main__":
    test_tiktok_uploader()
