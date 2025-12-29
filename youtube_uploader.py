"""
YouTube Automation System for Clipppy

Automatically uploads clips to YouTube with AI-generated metadata,
optimal scheduling, and performance tracking.
"""

import os
import json
import logging
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import yaml

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Google API imports
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    print("⚠️ YouTube API not available. Install with: pip install google-api-python-client google-auth-oauthlib")

# AI imports for metadata generation
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# YouTube API scopes
SCOPES = [
    'https://www.googleapis.com/auth/youtube.upload',
    'https://www.googleapis.com/auth/youtube',
    'https://www.googleapis.com/auth/youtube.readonly'
]


@dataclass
class VideoMetadata:
    """Metadata for a YouTube video"""
    title: str
    description: str
    tags: List[str]
    category_id: str = "20"  # Gaming category
    privacy_status: str = "public"  # public, private, unlisted
    made_for_kids: bool = False
    thumbnail_path: Optional[str] = None
    scheduled_time: Optional[datetime] = None


@dataclass
class UploadResult:
    """Result of a video upload"""
    success: bool
    video_id: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    scheduled: bool = False


class OptimalScheduler:
    """Determines posting times using simple interval-based scheduling"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_hours_between_posts = config.get('youtube', {}).get('min_hours_between_posts', 2)
        self.max_posts_per_day = config.get('youtube', {}).get('max_posts_per_day', 6)
    
    def get_next_optimal_slot(self, last_post_time: Optional[datetime] = None, queue_position: int = 0) -> datetime:
        """
        Calculate the next posting time using simple intervals.
        
        Args:
            last_post_time: When the last video was posted or scheduled
            queue_position: Position in queue (0 = first, 1 = second, etc.)
            
        Returns:
            Next datetime to post
        """
        now = datetime.now()
        
        # If we have a last post time, schedule after the interval
        if last_post_time:
            next_time = last_post_time + timedelta(hours=self.min_hours_between_posts)
            # If that's in the past, start from now
            if next_time < now:
                next_time = now + timedelta(minutes=5)  # Start in 5 minutes
        else:
            # No previous posts, start immediately (or in 5 minutes)
            next_time = now + timedelta(minutes=5)
        
        logger.info(f"📅 Next upload slot: {next_time.strftime('%A, %B %d at %I:%M %p')} (in {int((next_time - now).total_seconds() / 60)} minutes)")
        return next_time


class MetadataGenerator:
    """Generates AI-powered titles, descriptions, and tags for videos"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        
        # Prefer OpenAI over Anthropic (better for short-form content)
        if OPENAI_AVAILABLE and self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.ai_provider = 'openai'
            logger.info("✅ Using OpenAI for metadata generation")
        elif ANTHROPIC_AVAILABLE and self.anthropic_api_key:
            self.client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            self.model = "claude-3-5-sonnet-20240620"
            self.ai_provider = 'anthropic'
            logger.info("✅ Using Anthropic for metadata generation")
        else:
            self.ai_provider = None
            logger.info("ℹ️ No AI available, using templates")
    
    def generate_metadata(
        self,
        streamer_name: str,
        clip_context: Optional[Dict] = None,
        is_compilation: bool = False
    ) -> VideoMetadata:
        """
        Generate optimized metadata for a clip or compilation.
        
        Args:
            streamer_name: Name of the streamer
            clip_context: Optional context about the clip (viral signals, game, etc.)
            is_compilation: Whether this is a weekly compilation
            
        Returns:
            VideoMetadata with AI-generated content
        """
        if self.ai_provider and not is_compilation:
            # Use AI for individual clips
            if self.ai_provider == 'openai':
                return self._generate_with_openai(streamer_name, clip_context)
            else:
                return self._generate_with_anthropic(streamer_name, clip_context)
        else:
            # Use template-based for compilations or if AI unavailable
            return self._generate_with_template(streamer_name, clip_context, is_compilation)
    
    def _generate_with_openai(self, streamer_name: str, clip_context: Optional[Dict]) -> VideoMetadata:
        """Generate metadata using OpenAI GPT"""
        try:
            # Build context
            context_str = ""
            if clip_context:
                game = clip_context.get('game_name', 'Unknown')
                reason = clip_context.get('reason', '')
                viral_score = clip_context.get('viral_score', 0)
                context_str = f"\nGame: {game}\nClip Reason: {reason}\nViral Score: {viral_score:.2f}"
            
            prompt = f"""Generate YouTube metadata for a gaming clip from streamer "{streamer_name}".
{context_str}

Create:
1. A catchy, clickable title (50-60 characters, capitalize key words, use emojis)
2. An engaging description (150-200 words with relevant keywords)
3. 10-15 relevant tags

Make the title exciting using proven YouTube formats like:
- "This is INSANE! 🔥"
- "You WON'T believe this... 😱"
- "[Game] but [unexpected twist] 💀"
- "When [X] goes WRONG ⚡"

The description should:
- Hook viewers in the first line
- Include relevant gaming keywords
- Have a call-to-action (like/subscribe)
- Include streamer credit

Format your response as JSON:
{{"title": "...", "description": "...", "tags": ["tag1", "tag2", ...]}}"""

            # Use new OpenAI API (v1.0+)
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cheap for this task
                messages=[
                    {"role": "system", "content": "You are a YouTube metadata expert specializing in gaming content. Generate engaging, SEO-optimized titles and descriptions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.8  # More creative
            )
            
            # Parse response
            content = response.choices[0].message.content
            # Extract JSON from response (might be wrapped in markdown)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            metadata_dict = json.loads(content)
            
            logger.info(f"✅ Generated AI title: {metadata_dict['title']}")
            
            return VideoMetadata(
                title=metadata_dict['title'][:100],  # YouTube limit
                description=metadata_dict['description'],
                tags=metadata_dict['tags'][:15],  # YouTube allows up to 500 chars of tags
                category_id="20",  # Gaming
                privacy_status="public"
            )
            
        except Exception as e:
            logger.warning(f"⚠️ OpenAI generation failed: {e}, falling back to template")
            return self._generate_with_template(streamer_name, clip_context, False)
    
    def _generate_with_anthropic(self, streamer_name: str, clip_context: Optional[Dict]) -> VideoMetadata:
        """Generate metadata using Claude AI (fallback)"""
        try:
            # Build context prompt
            context_str = ""
            if clip_context:
                game = clip_context.get('game_name', 'Unknown')
                reason = clip_context.get('reason', '')
                viral_score = clip_context.get('viral_score', 0)
                context_str = f"\nGame: {game}\nClip Reason: {reason}\nViral Score: {viral_score:.2f}"
            
            prompt = f"""Generate YouTube metadata for a gaming clip from streamer "{streamer_name}".
{context_str}

Create:
1. A catchy, clickable title (50-60 characters, capitalize key words)
2. An engaging description (150-200 words with relevant keywords)
3. 10-15 relevant tags

Make the title exciting and use proven YouTube formats like:
- "This is INSANE!"
- "You WON'T believe this..."
- "[Game] but [unexpected twist]"
- "When [X] goes WRONG"

The description should:
- Hook viewers in the first line
- Include relevant gaming keywords
- Have a call-to-action (like/subscribe)
- Include streamer credit

Format your response as JSON:
{{"title": "...", "description": "...", "tags": ["tag1", "tag2", ...]}}"""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            content = response.content[0].text
            # Extract JSON from response (might be wrapped in markdown)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            metadata_dict = json.loads(content)
            
            return VideoMetadata(
                title=metadata_dict['title'][:100],  # YouTube limit
                description=metadata_dict['description'],
                tags=metadata_dict['tags'][:15],  # YouTube allows up to 500 chars of tags
                category_id="20",  # Gaming
                privacy_status="public"
            )
            
        except Exception as e:
            logger.warning(f"⚠️ AI generation failed: {e}, falling back to template")
            return self._generate_with_template(streamer_name, clip_context, False)
    
    def _generate_with_template(
        self,
        streamer_name: str,
        clip_context: Optional[Dict],
        is_compilation: bool
    ) -> VideoMetadata:
        """Generate metadata using templates"""
        
        if is_compilation:
            date_str = datetime.now().strftime("%B %d, %Y")
            title = f"{streamer_name} - THIS WEEK'S BEST MOMENTS! ({date_str})"
            description = f"""🎮 The BEST highlights from {streamer_name} this week!

Watch as {streamer_name} delivers non-stop entertainment with epic gameplay moments, hilarious reactions, and clutch plays.

📺 Watch {streamer_name} live: https://twitch.tv/{streamer_name.lower()}

⏰ New highlights every week!
👍 Like and Subscribe for more gaming content!

#gaming #{streamer_name.lower()} #highlights #funny #epic #gaming2024"""
            
            tags = [
                streamer_name,
                f"{streamer_name} highlights",
                "gaming",
                "funny moments",
                "epic plays",
                "twitch highlights",
                "best moments",
                "compilation",
                "gaming compilation"
            ]
        else:
            # Individual clip - generate unique title
            game = clip_context.get('game_name', 'Gaming') if clip_context else 'Gaming'
            viral_score = clip_context.get('viral_score', 0.25) if clip_context else 0.25
            reason = clip_context.get('reason', '') if clip_context else ''
            
            # Title templates for variety
            templates = [
                "{streamer} - {action} {game} {emoji}",
                "{streamer}'s {adjective} {game} Play! {emoji}",
                "WATCH: {streamer} {action} in {game} {emoji}",
                "{streamer} - {adjective} {game} Highlight {emoji}",
                "{game}: {streamer}'s {adjective} Move {emoji}",
                "{streamer} DESTROYS in {game} {emoji}",
                "INSANE {game} Play by {streamer} {emoji}",
                "{streamer}'s {adjective} {game} Moment {emoji}",
            ]
            
            actions = [
                "Goes OFF", "Clutches", "Dominates", "Wipes the Team",
                "Gets a HUGE Play", "Makes an Insane Move", "Pops Off",
                "Carries", "Outplays Everyone", "Gets the ACE", "Locks In"
            ]
            
            adjectives = [
                "INSANE", "EPIC", "CLUTCH", "CRAZY", "GODLIKE", 
                "LEGENDARY", "MASSIVE", "INCREDIBLE", "NASTY", "WILD"
            ]
            
            emojis = ["🔥", "⚡", "💀", "🎯", "💪", "👑", "🚀", "💥", "😱", "🔫"]
            
            # Use timestamp hash for consistent but varied selection
            import hashlib
            seed = int(hashlib.md5(f"{streamer_name}{time.time()}".encode()).hexdigest(), 16) % 10000
            
            template = templates[seed % len(templates)]
            title = template.format(
                streamer=streamer_name,
                action=actions[seed % len(actions)],
                adjective=adjectives[(seed + 1) % len(adjectives)],
                game=game,
                emoji=emojis[seed % len(emojis)]
            )
            
            # Ensure under 100 chars
            if len(title) > 95:
                title = title[:92] + "..."
            
            description = f"""🔥 {streamer_name} delivers an INSANE moment in {game}!

This is why {streamer_name} is one of the best streamers out there. Watch until the end!

📺 Watch {streamer_name} live: https://twitch.tv/{streamer_name.lower()}

👍 Like and Subscribe for more epic gaming moments!

#gaming #{streamer_name.lower()} #{game.lower().replace(' ', '')} #epic #viral #twitch"""
            
            tags = [
                streamer_name,
                game,
                "gaming",
                "epic moment",
                "viral",
                "twitch",
                "gaming highlights",
                f"{streamer_name} {game}",
                "insane play"
            ]
        
        return VideoMetadata(
            title=title[:100],
            description=description,
            tags=tags[:15],
            category_id="20",
            privacy_status="public"
        )


class YouTubeUploader:
    """Handles YouTube video uploads with OAuth authentication"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        self.credentials_file = Path("config/youtube_credentials.json")
        self.token_file = Path("config/youtube_token.pickle")
        
        self.youtube = None
        self.metadata_generator = MetadataGenerator(self.config)
        self.scheduler = OptimalScheduler(self.config)
        
        self.upload_queue_file = Path("data/youtube_upload_queue.json")
        self.upload_queue_file.parent.mkdir(exist_ok=True)
        
        # Get delete_after_upload setting
        self.delete_after_upload = self.config.get('youtube', {}).get('delete_after_upload', True)
        
        if not YOUTUBE_API_AVAILABLE:
            logger.error("❌ YouTube API libraries not installed!")
            return
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def authenticate(self) -> bool:
        """
        Authenticate with YouTube API using OAuth 2.0.
        
        Returns:
            True if authentication successful
        """
        if not YOUTUBE_API_AVAILABLE:
            logger.error("❌ YouTube API not available")
            return False
        
        creds = None
        
        # Load existing token if available
        if self.token_file.exists():
            with open(self.token_file, 'rb') as token:
                creds = pickle.load(token)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("🔄 Refreshing YouTube credentials...")
                creds.refresh(Request())
            else:
                if not self.credentials_file.exists():
                    logger.error(f"❌ Credentials file not found: {self.credentials_file}")
                    logger.error("   Get credentials from: https://console.cloud.google.com/")
                    return False
                
                logger.info("🔐 Starting YouTube OAuth flow...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_file),
                    SCOPES
                )
                creds = flow.run_local_server(port=8080)
            
            # Save credentials for next time
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
        
        # Build YouTube service
        try:
            self.youtube = build('youtube', 'v3', credentials=creds)
            logger.info("✅ YouTube authentication successful!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to build YouTube service: {e}")
            return False
    
    def upload_video(
        self,
        video_path: str,
        metadata: VideoMetadata,
        schedule: bool = True
    ) -> UploadResult:
        """
        Upload a video to YouTube.
        
        Args:
            video_path: Path to the video file
            metadata: Video metadata (title, description, tags, etc.)
            schedule: Whether to schedule for optimal time (or post immediately)
            
        Returns:
            UploadResult with video ID and URL
        """
        if not self.youtube:
            if not self.authenticate():
                return UploadResult(success=False, error="Authentication failed")
        
        video_path = Path(video_path)
        if not video_path.exists():
            return UploadResult(success=False, error=f"Video file not found: {video_path}")
        
        try:
            # Prepare video metadata
            body = {
                'snippet': {
                    'title': metadata.title,
                    'description': metadata.description,
                    'tags': metadata.tags,
                    'categoryId': metadata.category_id
                },
                'status': {
                    'privacyStatus': metadata.privacy_status,
                    'selfDeclaredMadeForKids': metadata.made_for_kids
                }
            }
            
            # Add scheduling if requested
            scheduled = False
            if schedule and metadata.scheduled_time:
                body['status']['publishAt'] = metadata.scheduled_time.isoformat() + 'Z'
                body['status']['privacyStatus'] = 'private'  # Must be private until publish time
                scheduled = True
            
            # Upload video
            logger.info(f"📤 Uploading video: {metadata.title}")
            media = MediaFileUpload(
                str(video_path),
                chunksize=1024*1024,  # 1MB chunks
                resumable=True
            )
            
            request = self.youtube.videos().insert(
                part='snippet,status',
                body=body,
                media_body=media
            )
            
            # Execute upload with progress
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logger.info(f"   Upload progress: {progress}%")
            
            video_id = response['id']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            logger.info(f"✅ Video uploaded successfully!")
            logger.info(f"   Video ID: {video_id}")
            logger.info(f"   URL: {video_url}")
            
            if scheduled:
                logger.info(f"   Scheduled for: {metadata.scheduled_time}")
            
            # Upload thumbnail if provided
            if metadata.thumbnail_path and Path(metadata.thumbnail_path).exists():
                try:
                    self.youtube.thumbnails().set(
                        videoId=video_id,
                        media_body=MediaFileUpload(metadata.thumbnail_path)
                    ).execute()
                    logger.info("✅ Thumbnail uploaded")
                except Exception as e:
                    logger.warning(f"⚠️ Thumbnail upload failed: {e}")
            
            return UploadResult(
                success=True,
                video_id=video_id,
                video_url=video_url,
                scheduled=scheduled
            )
            
        except HttpError as e:
            error_msg = f"YouTube API error: {e.resp.status} - {e.content}"
            logger.error(f"❌ Upload failed: {error_msg}")
            return UploadResult(success=False, error=error_msg)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Upload failed: {error_msg}")
            import traceback
            traceback.print_exc()
            return UploadResult(success=False, error=error_msg)
    
    def queue_video(
        self,
        video_path: str,
        streamer_name: str,
        clip_context: Optional[Dict] = None,
        is_compilation: bool = False,
        priority: int = 5
    ):
        """
        Add a video to the upload queue.
        
        Args:
            video_path: Path to video file
            streamer_name: Name of streamer
            clip_context: Optional context about the clip
            is_compilation: Whether this is a compilation video
            priority: Priority (1-10, 10 is highest)
        """
        # Load existing queue
        queue = self._load_queue()
        
        # Generate metadata
        metadata = self.metadata_generator.generate_metadata(
            streamer_name,
            clip_context,
            is_compilation
        )
        
        # Calculate next optimal posting time based on last scheduled video
        last_scheduled_time = self._get_last_scheduled_time(queue)
        scheduled_time = self.scheduler.get_next_optimal_slot(last_scheduled_time)
        metadata.scheduled_time = scheduled_time
        
        # Add to queue (convert datetime to ISO string for JSON serialization)
        metadata_dict = asdict(metadata)
        if metadata_dict.get('scheduled_time'):
            metadata_dict['scheduled_time'] = metadata_dict['scheduled_time'].isoformat()
        
        queue_item = {
            'video_path': str(video_path),
            'streamer_name': streamer_name,
            'metadata': metadata_dict,
            'is_compilation': is_compilation,
            'priority': priority,
            'queued_at': datetime.now().isoformat(),
            'status': 'pending',
            'scheduled_time': metadata_dict['scheduled_time']  # Add at top level for easier sorting
        }
        
        queue.append(queue_item)
        
        # Sort by priority (highest first) and scheduled time
        queue.sort(key=lambda x: (-x['priority'], x['scheduled_time']))
        
        # Save queue
        self._save_queue(queue)
        
        logger.info(f"📋 Added to upload queue: {metadata.title}")
        logger.info(f"   Scheduled for: {scheduled_time.strftime('%A, %B %d at %I:%M %p')}")
        logger.info(f"   Queue position: {len([q for q in queue if q['status'] == 'pending'])}")
    
    def process_queue(self, max_uploads: int = 1):
        """
        Process pending uploads from the queue.
        
        Args:
            max_uploads: Maximum number of videos to upload in this run
        """
        queue = self._load_queue()
        pending = [q for q in queue if q['status'] == 'pending']
        
        if not pending:
            logger.info("📋 Upload queue is empty")
            return
        
        logger.info(f"📋 Processing upload queue ({len(pending)} pending)")
        
        # Sort by priority and scheduled time to get the right order
        pending.sort(key=lambda x: (-x.get('priority', 5), x.get('scheduled_time', '9999-12-31')))
        
        uploaded = 0
        now = datetime.now()
        
        # Get the last ACTUAL upload time to calculate intervals from
        last_uploaded_time = self._get_last_uploaded_time(queue)
        
        for item in pending[:max_uploads]:
            # Check if enough time has passed since last upload
            if last_uploaded_time:
                time_since_last = (now - last_uploaded_time).total_seconds() / 3600  # hours
                min_interval = self.scheduler.min_hours_between_posts
                
                if time_since_last < min_interval:
                    wait_time = min_interval - time_since_last
                    next_time = last_uploaded_time + timedelta(hours=min_interval)
                    logger.info(f"⏰ Next upload ready in {int(wait_time * 60)} minutes (at {next_time.strftime('%I:%M %p')})")
                    break
            
            # Time to upload! Update scheduled time to now for logging
            item['scheduled_time'] = now.isoformat()
            if 'metadata' in item and isinstance(item['metadata'], dict):
                item['metadata']['scheduled_time'] = now.isoformat()
            
            # Upload video
            metadata = VideoMetadata(**item['metadata'])
            result = self.upload_video(
                item['video_path'],
                metadata,
                schedule=False  # Already scheduled
            )
            
            # Update queue item
            item['status'] = 'uploaded' if result.success else 'failed'
            item['uploaded_at'] = datetime.now().isoformat()
            if result.success:
                item['video_id'] = result.video_id
                item['video_url'] = result.video_url
                
                # Delete the video file after successful upload (if enabled)
                if self.delete_after_upload:
                    try:
                        video_file = Path(item['video_path'])
                        if video_file.exists():
                            video_file.unlink()
                            logger.info(f"🗑️ Deleted uploaded video file: {video_file.name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to delete video file: {e}")
                
                # Update last_uploaded_time for next iteration
                last_uploaded_time = datetime.now()
            else:
                item['error'] = result.error
            
            uploaded += 1
        
        # Save updated queue
        self._save_queue(queue)
        logger.info(f"✅ Processed {uploaded} upload(s)")
    
    def _load_queue(self) -> List[Dict]:
        """Load upload queue from file"""
        if self.upload_queue_file.exists():
            with open(self.upload_queue_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_queue(self, queue: List[Dict]):
        """Save upload queue to file"""
        with open(self.upload_queue_file, 'w') as f:
            json.dump(queue, f, indent=2)
    
    def _get_last_post_time(self, queue: List[Dict]) -> Optional[datetime]:
        """Get the time of the last uploaded video"""
        uploaded = [q for q in queue if q['status'] == 'uploaded' and 'uploaded_at' in q]
        if uploaded:
            last = max(uploaded, key=lambda x: x['uploaded_at'])
            return datetime.fromisoformat(last['uploaded_at'])
        return None
    
    def _get_last_uploaded_time(self, queue: List[Dict]) -> Optional[datetime]:
        """Get the actual upload time of the last uploaded video"""
        uploaded = [q for q in queue if q['status'] == 'uploaded' and 'uploaded_at' in q]
        if uploaded:
            last = max(uploaded, key=lambda x: x['uploaded_at'])
            return datetime.fromisoformat(last['uploaded_at'])
        return None
    
    def _get_last_scheduled_time(self, queue: List[Dict]) -> Optional[datetime]:
        """Get the scheduled time of the last pending or uploaded video"""
        # Check both pending and uploaded videos
        all_videos = [q for q in queue if q.get('scheduled_time')]
        if all_videos:
            # Find the latest scheduled time
            last = max(all_videos, key=lambda x: x['scheduled_time'])
            return datetime.fromisoformat(last['scheduled_time'])
        return None
    
    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        queue = self._load_queue()
        pending = [q for q in queue if q['status'] == 'pending']
        uploaded = [q for q in queue if q['status'] == 'uploaded']
        failed = [q for q in queue if q['status'] == 'failed']
        
        return {
            'total': len(queue),
            'pending': len(pending),
            'uploaded': len(uploaded),
            'failed': len(failed),
            'next_upload': pending[0] if pending else None
        }
    
    def reschedule_pending_videos(self):
        """
        Reschedule all pending videos using interval-based scheduling.
        Useful when switching from optimal slots to interval-based scheduling.
        """
        queue = self._load_queue()
        pending = [q for q in queue if q['status'] == 'pending']
        
        if not pending:
            logger.info("📋 No pending videos to reschedule")
            return
        
        logger.info(f"📅 Rescheduling {len(pending)} pending video(s) with {self.scheduler.min_hours_between_posts}h intervals...")
        
        # Start from now
        next_time = datetime.now() + timedelta(minutes=5)
        
        for item in pending:
            # Update scheduled time
            item['scheduled_time'] = next_time.isoformat()
            if 'metadata' in item and isinstance(item['metadata'], dict):
                item['metadata']['scheduled_time'] = next_time.isoformat()
            
            logger.info(f"   • {item['metadata']['title'][:50]}... → {next_time.strftime('%I:%M %p')}")
            
            # Next video scheduled after interval
            next_time += timedelta(hours=self.scheduler.min_hours_between_posts)
        
        # Sort by scheduled time
        queue.sort(key=lambda x: x.get('scheduled_time', '9999-12-31'))
        
        # Save updated queue
        self._save_queue(queue)
        logger.info(f"✅ Rescheduled {len(pending)} video(s)")


if __name__ == '__main__':
    # Test authentication
    uploader = YouTubeUploader()
    if uploader.authenticate():
        print("✅ YouTube authentication successful!")
        print("\n📊 Queue status:")
        status = uploader.get_queue_status()
        print(f"   Pending: {status['pending']}")
        print(f"   Uploaded: {status['uploaded']}")
        print(f"   Failed: {status['failed']}")
    else:
        print("❌ Authentication failed")

