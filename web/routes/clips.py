"""
Clips Routes

CRUD operations for clips, approval workflow, re-enhancement.
"""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..database import get_db
from ..models import User, Clip, ClipStatus
from ..schemas import ClipResponse, ClipUpdate, ClipApprove, ClipReEnhance
from ..auth import get_current_user_required

router = APIRouter(prefix="/api/clips", tags=["clips"])


@router.get("", response_model=List[ClipResponse])
async def list_clips(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """List user's clips with optional status filter"""
    query = db.query(Clip).filter(Clip.user_id == user.id)

    if status:
        try:
            clip_status = ClipStatus(status)
            query = query.filter(Clip.status == clip_status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    clips = query.order_by(desc(Clip.detected_at)).offset(offset).limit(limit).all()

    # Add computed URLs
    for clip in clips:
        clip.video_url = clip.clip_url
        clip.thumbnail_url = clip.thumbnail_path

    return clips


@router.get("/pending", response_model=List[ClipResponse])
async def list_pending_clips(
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Get clips awaiting review"""
    clips = db.query(Clip).filter(
        Clip.user_id == user.id,
        Clip.status == ClipStatus.PENDING_REVIEW
    ).order_by(desc(Clip.detected_at)).all()

    for clip in clips:
        clip.video_url = clip.clip_url
        clip.thumbnail_url = clip.thumbnail_path

    return clips


@router.get("/stats")
async def get_clip_stats(
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Get clip statistics for user"""
    total = db.query(Clip).filter(Clip.user_id == user.id).count()
    pending = db.query(Clip).filter(
        Clip.user_id == user.id,
        Clip.status == ClipStatus.PENDING_REVIEW
    ).count()
    posted = db.query(Clip).filter(
        Clip.user_id == user.id,
        Clip.status == ClipStatus.POSTED
    ).count()
    processing = db.query(Clip).filter(
        Clip.user_id == user.id,
        Clip.status == ClipStatus.PROCESSING
    ).count()

    # Today's clips
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_clips = db.query(Clip).filter(
        Clip.user_id == user.id,
        Clip.detected_at >= today_start
    ).count()

    return {
        "total": total,
        "pending_review": pending,
        "posted": posted,
        "processing": processing,
        "today": today_clips
    }


@router.get("/{clip_id}", response_model=ClipResponse)
async def get_clip(
    clip_id: str,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Get a specific clip"""
    clip = db.query(Clip).filter(
        Clip.clip_id == clip_id,
        Clip.user_id == user.id
    ).first()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    clip.video_url = clip.clip_url
    clip.thumbnail_url = clip.thumbnail_path
    return clip


@router.patch("/{clip_id}", response_model=ClipResponse)
async def update_clip(
    clip_id: str,
    update: ClipUpdate,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Update clip metadata (title, description, tags)"""
    clip = db.query(Clip).filter(
        Clip.clip_id == clip_id,
        Clip.user_id == user.id
    ).first()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    if update.final_title is not None:
        clip.final_title = update.final_title
    if update.final_description is not None:
        clip.final_description = update.final_description
    if update.final_tags is not None:
        clip.final_tags = update.final_tags

    db.commit()
    db.refresh(clip)

    clip.video_url = clip.clip_url
    clip.thumbnail_url = clip.thumbnail_path
    return clip


@router.post("/{clip_id}/approve", response_model=ClipResponse)
async def approve_clip(
    clip_id: str,
    approval: ClipApprove,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Approve a clip for posting"""
    clip = db.query(Clip).filter(
        Clip.clip_id == clip_id,
        Clip.user_id == user.id
    ).first()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    if clip.status != ClipStatus.PENDING_REVIEW:
        raise HTTPException(
            status_code=400,
            detail=f"Clip is not pending review (status: {clip.status.value})"
        )

    # Update with final metadata
    if approval.final_title:
        clip.final_title = approval.final_title
    if approval.final_description:
        clip.final_description = approval.final_description
    if approval.final_tags:
        clip.final_tags = approval.final_tags

    clip.status = ClipStatus.APPROVED
    clip.reviewed_at = datetime.utcnow()
    db.commit()

    # Queue upload task (if YouTube connected and enabled)
    if approval.post_to_youtube and user.youtube_connected:
        # Import here to avoid circular imports
        try:
            from ...celery_tasks import upload_to_youtube
            upload_to_youtube.delay(user.id, clip.id)
        except ImportError:
            # Celery not available, mark for manual processing
            pass

    db.refresh(clip)
    clip.video_url = clip.clip_url
    clip.thumbnail_url = clip.thumbnail_path
    return clip


@router.post("/{clip_id}/reject")
async def reject_clip(
    clip_id: str,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Reject/discard a clip"""
    clip = db.query(Clip).filter(
        Clip.clip_id == clip_id,
        Clip.user_id == user.id
    ).first()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    clip.status = ClipStatus.REJECTED
    clip.reviewed_at = datetime.utcnow()
    db.commit()

    return {"message": "Clip rejected", "clip_id": clip_id}


@router.post("/{clip_id}/reenhance", response_model=ClipResponse)
async def reenhance_clip(
    clip_id: str,
    settings: ClipReEnhance,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Re-enhance a clip with different settings"""
    clip = db.query(Clip).filter(
        Clip.clip_id == clip_id,
        Clip.user_id == user.id
    ).first()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    if clip.status not in [ClipStatus.PENDING_REVIEW, ClipStatus.REJECTED]:
        raise HTTPException(
            status_code=400,
            detail="Can only re-enhance clips that are pending review or rejected"
        )

    # Set back to processing
    clip.status = ClipStatus.PROCESSING
    clip.enhanced_at = None
    db.commit()

    # Queue re-enhancement task
    try:
        from ...celery_tasks import enhance_clip
        enhance_clip.delay(
            user.id,
            clip.id,
            preset=settings.enhancement_preset,
            caption_style=settings.caption_style,
            custom_settings=settings.custom_settings
        )
    except ImportError:
        # Celery not available
        clip.status = ClipStatus.PENDING_REVIEW
        db.commit()

    db.refresh(clip)
    clip.video_url = clip.clip_url
    clip.thumbnail_url = clip.thumbnail_path
    return clip


@router.delete("/{clip_id}")
async def delete_clip(
    clip_id: str,
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Permanently delete a clip"""
    clip = db.query(Clip).filter(
        Clip.clip_id == clip_id,
        Clip.user_id == user.id
    ).first()

    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    # TODO: Delete files from storage (local or S3)

    db.delete(clip)
    db.commit()

    return {"message": "Clip deleted", "clip_id": clip_id}
