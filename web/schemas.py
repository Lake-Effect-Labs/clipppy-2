"""
Pydantic Schemas for API Request/Response Validation
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


# Enums for API
class SubscriptionTierEnum(str, Enum):
    free = "free"
    creator = "creator"
    pro = "pro"


class ClipStatusEnum(str, Enum):
    processing = "processing"
    pending_review = "pending_review"
    approved = "approved"
    posted = "posted"
    rejected = "rejected"
    failed = "failed"


# User Schemas
class UserBase(BaseModel):
    twitch_username: str
    twitch_display_name: Optional[str] = None


class UserResponse(UserBase):
    id: int
    twitch_id: str
    twitch_profile_image: Optional[str] = None
    youtube_connected: bool = False
    youtube_channel_name: Optional[str] = None
    subscription_tier: SubscriptionTierEnum = SubscriptionTierEnum.free
    is_active: bool = True
    created_at: datetime

    class Config:
        from_attributes = True


class UserPublic(BaseModel):
    """Public user info (no sensitive data)"""
    id: int
    twitch_username: str
    twitch_display_name: Optional[str] = None
    twitch_profile_image: Optional[str] = None
    youtube_connected: bool = False

    class Config:
        from_attributes = True


# Settings Schemas
class UserSettingsBase(BaseModel):
    viral_threshold: float = Field(0.35, ge=0.1, le=1.0)
    min_unique_chatters: int = Field(25, ge=5, le=200)
    cooldown_seconds: int = Field(900, ge=60, le=7200)
    clips_per_stream: int = Field(10, ge=1, le=50)
    enhancement_preset: str = "energetic"
    caption_style: str = "default"
    include_watermark: bool = True
    watermark_text: Optional[str] = None
    auto_post_enabled: bool = False
    auto_post_threshold: float = Field(0.5, ge=0.1, le=1.0)
    post_to_youtube: bool = True
    post_to_tiktok: bool = False
    notify_on_clip: bool = True
    notify_email: bool = False
    custom_keywords: List[str] = []


class UserSettingsUpdate(UserSettingsBase):
    """For updating settings (all optional)"""
    viral_threshold: Optional[float] = None
    min_unique_chatters: Optional[int] = None
    cooldown_seconds: Optional[int] = None
    clips_per_stream: Optional[int] = None
    enhancement_preset: Optional[str] = None
    caption_style: Optional[str] = None
    include_watermark: Optional[bool] = None
    watermark_text: Optional[str] = None
    auto_post_enabled: Optional[bool] = None
    auto_post_threshold: Optional[float] = None
    post_to_youtube: Optional[bool] = None
    post_to_tiktok: Optional[bool] = None
    notify_on_clip: Optional[bool] = None
    notify_email: Optional[bool] = None
    custom_keywords: Optional[List[str]] = None


class UserSettingsResponse(UserSettingsBase):
    id: int
    user_id: int
    updated_at: datetime

    class Config:
        from_attributes = True


# Clip Schemas
class ClipBase(BaseModel):
    suggested_title: Optional[str] = None
    suggested_description: Optional[str] = None


class ClipResponse(BaseModel):
    id: int
    clip_id: str
    status: ClipStatusEnum
    viral_score: Optional[float] = None
    trigger_reason: Optional[str] = None
    stream_title: Optional[str] = None
    game_name: Optional[str] = None
    viewer_count: Optional[int] = None
    suggested_title: Optional[str] = None
    suggested_description: Optional[str] = None
    suggested_tags: Optional[List[str]] = None
    final_title: Optional[str] = None
    final_description: Optional[str] = None
    youtube_url: Optional[str] = None
    tiktok_url: Optional[str] = None
    detected_at: datetime
    enhanced_at: Optional[datetime] = None
    posted_at: Optional[datetime] = None
    thumbnail_url: Optional[str] = None
    video_url: Optional[str] = None

    class Config:
        from_attributes = True


class ClipUpdate(BaseModel):
    """For editing clip before posting"""
    final_title: Optional[str] = Field(None, max_length=200)
    final_description: Optional[str] = None
    final_tags: Optional[List[str]] = None


class ClipApprove(BaseModel):
    """For approving a clip for posting"""
    final_title: Optional[str] = None
    final_description: Optional[str] = None
    final_tags: Optional[List[str]] = None
    post_to_youtube: bool = True
    post_to_tiktok: bool = False


class ClipReEnhance(BaseModel):
    """Request to re-enhance with different settings"""
    enhancement_preset: str = "energetic"
    caption_style: Optional[str] = None
    custom_settings: Optional[dict] = None


# Admin Schemas
class AdminUserResponse(UserResponse):
    """Extended user info for admin"""
    twitch_email: Optional[str] = None
    is_admin: bool = False
    last_login_at: Optional[datetime] = None
    subscription_expires_at: Optional[datetime] = None
    clips_count: int = 0
    has_twitch_token: bool = False
    has_youtube_token: bool = False


class AdminStatsResponse(BaseModel):
    total_users: int
    active_users: int
    live_now: int
    clips_today: int
    clips_pending_review: int
    clips_posted_today: int


class ListenerStatus(BaseModel):
    user_id: int
    twitch_username: str
    started_at: datetime
    clips_this_session: int
    current_viral_score: float
    status: str  # running, idle, error


# Auth Schemas
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[int] = None
    twitch_id: Optional[str] = None
