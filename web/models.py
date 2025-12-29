"""
Database Models for Clipppy

User: A streamer using the platform
Clip: A viral clip detected and processed
UserSettings: User's preferences for clip detection and enhancement
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, Text, JSON, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
import enum

from .database import Base


class SubscriptionTier(enum.Enum):
    FREE = "free"
    CREATOR = "creator"
    PRO = "pro"
    ADMIN = "admin"


class ClipStatus(enum.Enum):
    PROCESSING = "processing"      # Being enhanced
    PENDING_REVIEW = "pending_review"  # Ready for user review
    APPROVED = "approved"          # User approved, ready to post
    POSTED = "posted"              # Uploaded to YouTube/TikTok
    REJECTED = "rejected"          # User discarded
    FAILED = "failed"              # Processing failed


class User(Base):
    """A streamer using Clipppy"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    # Twitch identity (primary login)
    twitch_id = Column(String(50), unique=True, index=True, nullable=False)
    twitch_username = Column(String(100), nullable=False)
    twitch_display_name = Column(String(100))
    twitch_profile_image = Column(String(500))
    twitch_email = Column(String(255))

    # Encrypted OAuth tokens
    twitch_token_encrypted = Column(Text)  # Encrypted JSON: {access_token, refresh_token, expires_at}
    youtube_token_encrypted = Column(Text)  # Encrypted JSON: {access_token, refresh_token, expires_at}

    # YouTube connection status
    youtube_connected = Column(Boolean, default=False)
    youtube_channel_id = Column(String(100))
    youtube_channel_name = Column(String(200))

    # Subscription
    subscription_tier = Column(SQLEnum(SubscriptionTier), default=SubscriptionTier.FREE)
    subscription_expires_at = Column(DateTime)
    stripe_customer_id = Column(String(100))

    # Account status
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login_at = Column(DateTime)

    # Relationships
    clips = relationship("Clip", back_populates="user", cascade="all, delete-orphan")
    settings = relationship("UserSettings", back_populates="user", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.twitch_username}>"

    @property
    def is_subscribed(self) -> bool:
        """Check if user has active paid subscription"""
        if self.subscription_tier == SubscriptionTier.FREE:
            return False
        if self.subscription_expires_at and self.subscription_expires_at < datetime.utcnow():
            return False
        return True


class UserSettings(Base):
    """User's preferences for clip detection and enhancement"""
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)

    # Viral detection settings
    viral_threshold = Column(Float, default=0.35)
    min_unique_chatters = Column(Integer, default=25)
    cooldown_seconds = Column(Integer, default=900)  # 15 minutes
    clips_per_stream = Column(Integer, default=10)  # Max clips per stream

    # Enhancement settings
    enhancement_preset = Column(String(50), default="energetic")  # energetic, chill, hype, minimal
    caption_style = Column(String(50), default="default")
    include_watermark = Column(Boolean, default=True)
    watermark_text = Column(String(100))  # Defaults to @username_clippy

    # Auto-posting settings
    auto_post_enabled = Column(Boolean, default=False)
    auto_post_threshold = Column(Float, default=0.5)  # Only auto-post if score >= this
    post_to_youtube = Column(Boolean, default=True)
    post_to_tiktok = Column(Boolean, default=False)

    # Notification preferences
    notify_on_clip = Column(Boolean, default=True)
    notify_email = Column(Boolean, default=False)

    # Advanced settings (JSON for flexibility)
    custom_keywords = Column(JSON, default=list)  # Custom emphasis words
    custom_config = Column(JSON, default=dict)  # Any other custom settings

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="settings")

    def __repr__(self):
        return f"<UserSettings user_id={self.user_id}>"

    def to_viral_config(self) -> dict:
        """Convert to config dict for ViralDetector"""
        return {
            "viral_algorithm": {
                "score_threshold": self.viral_threshold,
                "min_unique_chatters": self.min_unique_chatters,
                "cooldown_seconds": self.cooldown_seconds,
            },
            "enhancement": {
                "preset": self.enhancement_preset,
                "caption_style": self.caption_style,
                "watermark": {
                    "enabled": self.include_watermark,
                    "handle_text": self.watermark_text,
                }
            },
            "custom_keywords": self.custom_keywords or [],
        }


class Clip(Base):
    """A viral clip detected and processed"""
    __tablename__ = "clips"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Clip identification
    clip_id = Column(String(100), unique=True, index=True)  # Our internal ID
    twitch_clip_id = Column(String(100))  # Twitch's clip ID
    twitch_clip_url = Column(String(500))

    # Status
    status = Column(SQLEnum(ClipStatus), default=ClipStatus.PROCESSING)

    # Viral detection data
    viral_score = Column(Float)
    trigger_reason = Column(Text)  # Why it triggered
    detection_breakdown = Column(JSON)  # Full score breakdown

    # Stream context
    stream_title = Column(String(500))
    game_name = Column(String(200))
    viewer_count = Column(Integer)

    # File locations
    raw_clip_path = Column(String(500))  # Original clip
    enhanced_clip_path = Column(String(500))  # Enhanced version
    thumbnail_path = Column(String(500))

    # S3 paths (if using cloud storage)
    s3_raw_key = Column(String(500))
    s3_enhanced_key = Column(String(500))

    # Generated metadata
    suggested_title = Column(String(200))
    suggested_description = Column(Text)
    suggested_tags = Column(JSON)

    # User edits
    final_title = Column(String(200))
    final_description = Column(Text)
    final_tags = Column(JSON)

    # Posting results
    youtube_video_id = Column(String(50))
    youtube_url = Column(String(200))
    tiktok_video_id = Column(String(50))
    tiktok_url = Column(String(200))
    posted_at = Column(DateTime)

    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow)
    enhanced_at = Column(DateTime)
    reviewed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Processing metadata
    processing_time_seconds = Column(Float)
    error_message = Column(Text)

    # Relationship
    user = relationship("User", back_populates="clips")

    def __repr__(self):
        return f"<Clip {self.clip_id} status={self.status.value}>"

    @property
    def display_title(self) -> str:
        """Get the title to display (user edit or suggested)"""
        return self.final_title or self.suggested_title or "Untitled Clip"

    @property
    def clip_url(self) -> str:
        """Get the best available clip URL"""
        return self.enhanced_clip_path or self.raw_clip_path or self.twitch_clip_url


class StreamSession(Base):
    """Track active streaming sessions for analytics"""
    __tablename__ = "stream_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Session info
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    duration_seconds = Column(Integer)

    # Stats
    clips_created = Column(Integer, default=0)
    clips_approved = Column(Integer, default=0)
    clips_rejected = Column(Integer, default=0)
    peak_viewers = Column(Integer)
    avg_chat_velocity = Column(Float)

    # Stream metadata
    stream_title = Column(String(500))
    game_name = Column(String(200))

    def __repr__(self):
        return f"<StreamSession user_id={self.user_id} started={self.started_at}>"
