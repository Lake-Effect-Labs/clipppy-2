"""
Multi-Tenant Celery Tasks

User-aware tasks for clip creation, enhancement, and uploading.
Works alongside existing celery_tasks.py for backward compatibility.
"""
import os
import sys
import uuid
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def create_and_enhance_clip(self, user_id: int, clip_data: dict):
    """
    Create a clip via Twitch API and enhance it.

    Args:
        user_id: Database user ID
        clip_data: Dict with viral_score, reason, breakdown, etc.
    """
    from web.database import SessionLocal
    from web.models import User, Clip, ClipStatus, UserSettings
    from web.encryption import decrypt_token

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            logger.error(f"User {user_id} not found")
            return {"error": "User not found"}

        settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()

        # Create clip record
        clip_id = str(uuid.uuid4())[:8]
        clip = Clip(
            user_id=user_id,
            clip_id=clip_id,
            status=ClipStatus.PROCESSING,
            viral_score=clip_data.get("viral_score"),
            trigger_reason=clip_data.get("reason"),
            detection_breakdown=clip_data.get("breakdown"),
            detected_at=datetime.utcnow(),
        )
        db.add(clip)
        db.commit()

        logger.info(f"Created clip {clip_id} for user {user.twitch_username}")

        # Get user's Twitch token for clip creation
        if user.twitch_token_encrypted:
            try:
                twitch_tokens = decrypt_token(user.twitch_token_encrypted)
                access_token = twitch_tokens.get("access_token")

                # Create clip via Twitch API
                twitch_clip_url = create_twitch_clip(
                    broadcaster_id=user.twitch_id,
                    access_token=access_token,
                )

                if twitch_clip_url:
                    clip.twitch_clip_url = twitch_clip_url
                    db.commit()

                    # Download and enhance
                    enhance_clip.delay(user_id, clip.id)
                else:
                    # Clip creation failed, mark as pending for manual review
                    clip.status = ClipStatus.PENDING_REVIEW
                    clip.suggested_title = f"Viral moment (score: {clip_data.get('viral_score', 0):.2f})"
                    db.commit()

            except Exception as e:
                logger.error(f"Error creating Twitch clip: {e}")
                clip.status = ClipStatus.FAILED
                clip.error_message = str(e)
                db.commit()
        else:
            # No token, can't create clip
            clip.status = ClipStatus.FAILED
            clip.error_message = "No Twitch token available"
            db.commit()

        return {"clip_id": clip_id, "status": clip.status.value}

    except Exception as e:
        logger.error(f"Error in create_and_enhance_clip: {e}")
        raise self.retry(exc=e, countdown=60)
    finally:
        db.close()


def create_twitch_clip(broadcaster_id: str, access_token: str) -> str:
    """Create a clip via Twitch API"""
    import httpx
    from web.config import TWITCH_CLIENT_ID

    try:
        response = httpx.post(
            "https://api.twitch.tv/helix/clips",
            params={"broadcaster_id": broadcaster_id},
            headers={
                "Authorization": f"Bearer {access_token}",
                "Client-Id": TWITCH_CLIENT_ID,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("data"):
            clip_id = data["data"][0]["id"]
            return f"https://clips.twitch.tv/{clip_id}"

    except Exception as e:
        logger.error(f"Twitch clip creation failed: {e}")

    return None


@shared_task(bind=True, max_retries=2)
def enhance_clip(
    self,
    user_id: int,
    clip_db_id: int,
    preset: str = None,
    caption_style: str = None,
    custom_settings: dict = None,
):
    """
    Download and enhance a clip.

    Args:
        user_id: Database user ID
        clip_db_id: Clip database ID
        preset: Enhancement preset override
        caption_style: Caption style override
        custom_settings: Custom settings override
    """
    from web.database import SessionLocal
    from web.models import User, Clip, ClipStatus, UserSettings

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        clip = db.query(Clip).filter(Clip.id == clip_db_id).first()

        if not user or not clip:
            logger.error(f"User or clip not found: user={user_id}, clip={clip_db_id}")
            return {"error": "Not found"}

        settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()

        # Determine enhancement settings
        enhancement_preset = preset or (settings.enhancement_preset if settings else "energetic")

        logger.info(f"Enhancing clip {clip.clip_id} with preset: {enhancement_preset}")

        # Download clip if needed
        if clip.twitch_clip_url and not clip.raw_clip_path:
            raw_path = download_clip(clip.twitch_clip_url, user.twitch_username, clip.clip_id)
            if raw_path:
                clip.raw_clip_path = raw_path
                db.commit()

        # Enhance clip
        if clip.raw_clip_path:
            try:
                from clip_enhancer_v2 import ClipEnhancerV2

                enhancer = ClipEnhancerV2()
                enhanced_path = enhancer.enhance(
                    clip.raw_clip_path,
                    preset=enhancement_preset,
                    streamer_name=user.twitch_username,
                )

                if enhanced_path:
                    clip.enhanced_clip_path = enhanced_path
                    clip.enhanced_at = datetime.utcnow()
                    clip.status = ClipStatus.PENDING_REVIEW

                    # Generate suggested title
                    clip.suggested_title = generate_title(user.twitch_username, clip.viral_score)
                    clip.suggested_description = generate_description(user.twitch_username)

                    db.commit()
                    logger.info(f"Enhancement complete: {clip.clip_id}")

                    # TODO: Send notification to user

            except Exception as e:
                logger.error(f"Enhancement failed: {e}")
                clip.status = ClipStatus.FAILED
                clip.error_message = str(e)
                db.commit()
        else:
            # No raw clip to enhance, mark as pending for manual review
            clip.status = ClipStatus.PENDING_REVIEW
            clip.suggested_title = f"Viral moment (score: {clip.viral_score:.2f})"
            db.commit()

        return {"clip_id": clip.clip_id, "status": clip.status.value}

    except Exception as e:
        logger.error(f"Error in enhance_clip: {e}")
        raise self.retry(exc=e, countdown=120)
    finally:
        db.close()


def download_clip(clip_url: str, username: str, clip_id: str) -> str:
    """Download a Twitch clip using yt-dlp"""
    import subprocess
    from web.config import CLIPS_DIR

    output_dir = CLIPS_DIR / username / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"raw_{username}_{clip_id}.mp4"

    try:
        result = subprocess.run(
            ["yt-dlp", "-o", str(output_path), clip_url],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0 and output_path.exists():
            logger.info(f"Downloaded clip to {output_path}")
            return str(output_path)

    except Exception as e:
        logger.error(f"Clip download failed: {e}")

    return None


def generate_title(username: str, score: float) -> str:
    """Generate a suggested title for the clip"""
    import random

    templates = [
        f"{username} - INSANE Moment!",
        f"This is WHY you watch {username}...",
        f"{username} Goes OFF!",
        f"You WON'T believe this {username} play!",
        f"{username}'s BEST moment today!",
    ]
    return random.choice(templates)


def generate_description(username: str) -> str:
    """Generate a suggested description"""
    return f"""Watch this insane moment from {username}!

Watch {username} live: https://twitch.tv/{username.lower()}

Like and subscribe for more epic clips!

#gaming #{username.lower()} #twitch #clips #viral"""


@shared_task(bind=True, max_retries=3)
def upload_to_youtube(self, user_id: int, clip_db_id: int):
    """
    Upload an approved clip to YouTube.

    Args:
        user_id: Database user ID
        clip_db_id: Clip database ID
    """
    from web.database import SessionLocal
    from web.models import User, Clip, ClipStatus
    from web.encryption import decrypt_token

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        clip = db.query(Clip).filter(Clip.id == clip_db_id).first()

        if not user or not clip:
            return {"error": "Not found"}

        if not user.youtube_connected or not user.youtube_token_encrypted:
            return {"error": "YouTube not connected"}

        if clip.status != ClipStatus.APPROVED:
            return {"error": f"Clip not approved (status: {clip.status.value})"}

        video_path = clip.enhanced_clip_path or clip.raw_clip_path
        if not video_path or not Path(video_path).exists():
            return {"error": "No video file"}

        logger.info(f"Uploading clip {clip.clip_id} to YouTube for {user.twitch_username}")

        # Get YouTube credentials
        youtube_tokens = decrypt_token(user.youtube_token_encrypted)

        # Upload using existing youtube_uploader
        from youtube_uploader import YouTubeUploader, VideoMetadata

        uploader = YouTubeUploader()

        # Build credentials from stored tokens
        from google.oauth2.credentials import Credentials

        creds = Credentials(
            token=youtube_tokens.get("access_token"),
            refresh_token=youtube_tokens.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.getenv("YOUTUBE_CLIENT_ID"),
            client_secret=os.getenv("YOUTUBE_CLIENT_SECRET"),
        )

        # Set up uploader with user's credentials
        from googleapiclient.discovery import build
        uploader.youtube = build('youtube', 'v3', credentials=creds)

        # Build metadata
        metadata = VideoMetadata(
            title=clip.final_title or clip.suggested_title or clip.display_title,
            description=clip.final_description or clip.suggested_description or "",
            tags=clip.final_tags or clip.suggested_tags or [],
            category_id="20",  # Gaming
            privacy_status="public",
        )

        # Upload
        result = uploader.upload_video(video_path, metadata, schedule=False)

        if result.success:
            clip.youtube_video_id = result.video_id
            clip.youtube_url = result.video_url
            clip.status = ClipStatus.POSTED
            clip.posted_at = datetime.utcnow()
            db.commit()

            logger.info(f"Uploaded to YouTube: {result.video_url}")

            # Optionally delete local file
            # Path(video_path).unlink()

            return {"youtube_url": result.video_url}
        else:
            clip.error_message = result.error
            db.commit()
            return {"error": result.error}

    except Exception as e:
        logger.error(f"YouTube upload failed: {e}")
        raise self.retry(exc=e, countdown=300)
    finally:
        db.close()
