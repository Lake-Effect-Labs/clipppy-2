"""
Admin Routes

Admin dashboard, user management, system monitoring.
"""
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from ..database import get_db
from ..models import User, Clip, UserSettings, ClipStatus, StreamSession
from ..schemas import AdminUserResponse, AdminStatsResponse, ListenerStatus
from ..auth import get_admin_user
from ..encryption import mask_token, decrypt_token

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/stats", response_model=AdminStatsResponse)
async def get_admin_stats(
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get system-wide statistics"""
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()

    # Today's activity
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    clips_today = db.query(Clip).filter(Clip.detected_at >= today_start).count()
    clips_pending = db.query(Clip).filter(Clip.status == ClipStatus.PENDING_REVIEW).count()
    clips_posted_today = db.query(Clip).filter(
        Clip.status == ClipStatus.POSTED,
        Clip.posted_at >= today_start
    ).count()

    # TODO: Get actual live count from orchestrator
    live_now = 0

    return AdminStatsResponse(
        total_users=total_users,
        active_users=active_users,
        live_now=live_now,
        clips_today=clips_today,
        clips_pending_review=clips_pending,
        clips_posted_today=clips_posted_today
    )


@router.get("/users")
async def list_users(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: Optional[str] = None,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """List all users with admin details"""
    query = db.query(User)

    if search:
        query = query.filter(
            User.twitch_username.ilike(f"%{search}%") |
            User.twitch_email.ilike(f"%{search}%")
        )

    users = query.order_by(desc(User.created_at)).offset(offset).limit(limit).all()

    result = []
    for user in users:
        clips_count = db.query(Clip).filter(Clip.user_id == user.id).count()
        result.append({
            "id": user.id,
            "twitch_id": user.twitch_id,
            "twitch_username": user.twitch_username,
            "twitch_display_name": user.twitch_display_name,
            "twitch_email": user.twitch_email,
            "twitch_profile_image": user.twitch_profile_image,
            "youtube_connected": user.youtube_connected,
            "youtube_channel_name": user.youtube_channel_name,
            "subscription_tier": user.subscription_tier.value,
            "is_active": user.is_active,
            "is_admin": user.is_admin,
            "created_at": user.created_at,
            "last_login_at": user.last_login_at,
            "clips_count": clips_count,
            "has_twitch_token": bool(user.twitch_token_encrypted),
            "has_youtube_token": bool(user.youtube_token_encrypted),
        })

    return result


@router.get("/users/{user_id}")
async def get_user_detail(
    user_id: int,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get detailed user info for admin"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()

    # Get clip stats
    clips_total = db.query(Clip).filter(Clip.user_id == user_id).count()
    clips_pending = db.query(Clip).filter(
        Clip.user_id == user_id,
        Clip.status == ClipStatus.PENDING_REVIEW
    ).count()
    clips_posted = db.query(Clip).filter(
        Clip.user_id == user_id,
        Clip.status == ClipStatus.POSTED
    ).count()

    # Recent clips
    recent_clips = db.query(Clip).filter(
        Clip.user_id == user_id
    ).order_by(desc(Clip.detected_at)).limit(10).all()

    return {
        "user": {
            "id": user.id,
            "twitch_id": user.twitch_id,
            "twitch_username": user.twitch_username,
            "twitch_display_name": user.twitch_display_name,
            "twitch_email": user.twitch_email,
            "twitch_profile_image": user.twitch_profile_image,
            "youtube_connected": user.youtube_connected,
            "youtube_channel_id": user.youtube_channel_id,
            "youtube_channel_name": user.youtube_channel_name,
            "subscription_tier": user.subscription_tier.value,
            "is_active": user.is_active,
            "is_admin": user.is_admin,
            "created_at": user.created_at,
            "last_login_at": user.last_login_at,
            "has_twitch_token": bool(user.twitch_token_encrypted),
            "has_youtube_token": bool(user.youtube_token_encrypted),
        },
        "settings": {
            "viral_threshold": settings.viral_threshold if settings else 0.35,
            "min_unique_chatters": settings.min_unique_chatters if settings else 25,
            "cooldown_seconds": settings.cooldown_seconds if settings else 900,
            "enhancement_preset": settings.enhancement_preset if settings else "energetic",
            "auto_post_enabled": settings.auto_post_enabled if settings else False,
        } if settings else None,
        "stats": {
            "clips_total": clips_total,
            "clips_pending": clips_pending,
            "clips_posted": clips_posted,
        },
        "recent_clips": [
            {
                "id": clip.id,
                "clip_id": clip.clip_id,
                "status": clip.status.value,
                "viral_score": clip.viral_score,
                "detected_at": clip.detected_at,
                "suggested_title": clip.suggested_title,
            }
            for clip in recent_clips
        ]
    }


@router.post("/users/{user_id}/toggle-active")
async def toggle_user_active(
    user_id: int,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Enable/disable a user account"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot disable your own account")

    user.is_active = not user.is_active
    db.commit()

    return {"message": f"User {'enabled' if user.is_active else 'disabled'}", "is_active": user.is_active}


@router.post("/users/{user_id}/toggle-admin")
async def toggle_user_admin(
    user_id: int,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Grant/revoke admin access"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot modify your own admin status")

    user.is_admin = not user.is_admin
    db.commit()

    return {"message": f"Admin access {'granted' if user.is_admin else 'revoked'}", "is_admin": user.is_admin}


@router.get("/users/{user_id}/impersonate")
async def get_impersonation_token(
    user_id: int,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Get a token to view as a specific user (for debugging)"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    from ..auth import create_access_token
    token = create_access_token(
        data={"sub": str(user.id), "impersonated_by": admin.id},
        expires_delta=timedelta(hours=1)  # Short-lived for security
    )

    return {
        "message": f"Impersonation token for {user.twitch_username}",
        "token": token,
        "expires_in": 3600,
        "warning": "This token allows full access as this user. Use responsibly."
    }


@router.get("/listeners")
async def get_active_listeners(
    admin: User = Depends(get_admin_user),
):
    """Get status of all active listeners"""
    # TODO: Connect to orchestrator to get real listener status
    # For now, return placeholder
    return {
        "listeners": [],
        "message": "Orchestrator integration pending"
    }


@router.get("/logs")
async def get_system_logs(
    user_id: Optional[int] = None,
    level: str = Query("INFO", regex="^(DEBUG|INFO|WARNING|ERROR)$"),
    limit: int = Query(100, ge=1, le=1000),
    admin: User = Depends(get_admin_user),
):
    """Get system logs (placeholder - would connect to log aggregator)"""
    # TODO: Integrate with actual log system
    return {
        "logs": [],
        "message": "Log aggregation integration pending"
    }


@router.get("/health")
async def health_check(
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """System health check"""
    health = {
        "status": "healthy",
        "components": {}
    }

    # Database check
    try:
        db.execute("SELECT 1")
        health["components"]["database"] = {"status": "healthy"}
    except Exception as e:
        health["components"]["database"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "degraded"

    # TODO: Check Redis, Celery workers, etc.
    health["components"]["redis"] = {"status": "unknown", "message": "Not implemented"}
    health["components"]["celery"] = {"status": "unknown", "message": "Not implemented"}
    health["components"]["orchestrator"] = {"status": "unknown", "message": "Not implemented"}

    return health
