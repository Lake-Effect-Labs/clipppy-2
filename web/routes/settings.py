"""
Settings Routes

User settings CRUD operations.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User, UserSettings
from ..schemas import UserSettingsResponse, UserSettingsUpdate
from ..auth import get_current_user_required

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("", response_model=UserSettingsResponse)
async def get_settings(
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Get current user's settings"""
    settings = db.query(UserSettings).filter(UserSettings.user_id == user.id).first()

    if not settings:
        # Create default settings if none exist
        settings = UserSettings(
            user_id=user.id,
            watermark_text=f"@{user.twitch_username}_clippy"
        )
        db.add(settings)
        db.commit()
        db.refresh(settings)

    return settings


@router.patch("", response_model=UserSettingsResponse)
async def update_settings(
    update: UserSettingsUpdate,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Update user's settings"""
    settings = db.query(UserSettings).filter(UserSettings.user_id == user.id).first()

    if not settings:
        settings = UserSettings(user_id=user.id)
        db.add(settings)

    # Update only provided fields
    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if value is not None:
            setattr(settings, field, value)

    db.commit()
    db.refresh(settings)

    return settings


@router.post("/reset")
async def reset_settings(
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Reset settings to defaults"""
    settings = db.query(UserSettings).filter(UserSettings.user_id == user.id).first()

    if settings:
        # Reset to defaults
        settings.viral_threshold = 0.35
        settings.min_unique_chatters = 25
        settings.cooldown_seconds = 900
        settings.clips_per_stream = 10
        settings.enhancement_preset = "energetic"
        settings.caption_style = "default"
        settings.include_watermark = True
        settings.watermark_text = f"@{user.twitch_username}_clippy"
        settings.auto_post_enabled = False
        settings.auto_post_threshold = 0.5
        settings.post_to_youtube = True
        settings.post_to_tiktok = False
        settings.notify_on_clip = True
        settings.notify_email = False
        settings.custom_keywords = []

        db.commit()
        db.refresh(settings)

    return {"message": "Settings reset to defaults"}


@router.get("/presets")
async def get_enhancement_presets():
    """Get available enhancement presets"""
    return {
        "presets": [
            {
                "id": "energetic",
                "name": "Energetic",
                "description": "High energy with zoom effects, shake, and bold captions. Great for hype moments."
            },
            {
                "id": "chill",
                "name": "Chill",
                "description": "Subtle effects, smooth transitions. Good for chill streams and talking content."
            },
            {
                "id": "hype",
                "name": "Hype / MrBeast Style",
                "description": "Maximum effects, large captions, frequent zooms. For viral potential."
            },
            {
                "id": "minimal",
                "name": "Minimal",
                "description": "Clean look with minimal effects. Just captions and basic formatting."
            },
            {
                "id": "gaming",
                "name": "Gaming Focus",
                "description": "Optimized for gameplay clips. Less face focus, more gameplay emphasis."
            }
        ],
        "caption_styles": [
            {"id": "default", "name": "Default", "description": "Standard white text with black outline"},
            {"id": "bold", "name": "Bold", "description": "Larger, bolder text"},
            {"id": "subtle", "name": "Subtle", "description": "Smaller, less intrusive captions"},
            {"id": "colorful", "name": "Colorful", "description": "Emphasis words in different colors"},
        ]
    }
