"""
Clipppy Web Application

FastAPI-based web interface for multi-tenant clip management.
"""
from pathlib import Path
from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .config import APP_NAME, TEMPLATES_DIR, STATIC_DIR
from .database import init_db, get_db
from .models import User, Clip, ClipStatus
from .auth import get_current_user

# Import routes
from .routes import auth, clips, settings, admin

# Create FastAPI app
app = FastAPI(
    title=APP_NAME,
    description="Automated viral clip detection and posting for Twitch streamers",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Include API routers
app.include_router(auth.router)
app.include_router(clips.router)
app.include_router(settings.router)
app.include_router(admin.router)


@app.on_event("startup")
async def startup():
    """Initialize database on startup"""
    init_db()


# ============ HTML Pages ============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, user: User = Depends(get_current_user)):
    """Home/landing page"""
    if user:
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Main dashboard"""
    if not user:
        return RedirectResponse(url="/")

    # Get pending clips
    pending_clips = db.query(Clip).filter(
        Clip.user_id == user.id,
        Clip.status == ClipStatus.PENDING_REVIEW
    ).order_by(Clip.detected_at.desc()).all()

    # Get recent posted clips
    posted_clips = db.query(Clip).filter(
        Clip.user_id == user.id,
        Clip.status == ClipStatus.POSTED
    ).order_by(Clip.posted_at.desc()).limit(5).all()

    # Get stats
    stats = {
        "pending": len(pending_clips),
        "posted": db.query(Clip).filter(
            Clip.user_id == user.id,
            Clip.status == ClipStatus.POSTED
        ).count(),
        "total": db.query(Clip).filter(Clip.user_id == user.id).count(),
    }

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "pending_clips": pending_clips,
        "posted_clips": posted_clips,
        "stats": stats,
    })


@app.get("/clips", response_class=HTMLResponse)
async def clips_page(
    request: Request,
    status: str = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Clips list page"""
    if not user:
        return RedirectResponse(url="/")

    query = db.query(Clip).filter(Clip.user_id == user.id)

    if status:
        try:
            clip_status = ClipStatus(status)
            query = query.filter(Clip.status == clip_status)
        except ValueError:
            pass

    all_clips = query.order_by(Clip.detected_at.desc()).limit(100).all()

    return templates.TemplateResponse("clips.html", {
        "request": request,
        "user": user,
        "clips": all_clips,
        "current_status": status,
    })


@app.get("/clips/{clip_id}", response_class=HTMLResponse)
async def clip_detail(
    request: Request,
    clip_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Single clip detail/edit page"""
    if not user:
        return RedirectResponse(url="/")

    clip = db.query(Clip).filter(
        Clip.clip_id == clip_id,
        Clip.user_id == user.id
    ).first()

    if not clip:
        return RedirectResponse(url="/clips")

    return templates.TemplateResponse("clip_detail.html", {
        "request": request,
        "user": user,
        "clip": clip,
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Settings page"""
    if not user:
        return RedirectResponse(url="/")

    from .models import UserSettings
    user_settings = db.query(UserSettings).filter(UserSettings.user_id == user.id).first()

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "user": user,
        "settings": user_settings,
    })


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Admin dashboard"""
    if not user or not user.is_admin:
        return RedirectResponse(url="/dashboard")

    # Get stats
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    total_clips = db.query(Clip).count()
    pending_clips = db.query(Clip).filter(Clip.status == ClipStatus.PENDING_REVIEW).count()

    # Get recent users
    recent_users = db.query(User).order_by(User.created_at.desc()).limit(10).all()

    return templates.TemplateResponse("admin/index.html", {
        "request": request,
        "user": user,
        "stats": {
            "total_users": total_users,
            "active_users": active_users,
            "total_clips": total_clips,
            "pending_clips": pending_clips,
        },
        "recent_users": recent_users,
    })


@app.get("/admin/users/{user_id}", response_class=HTMLResponse)
async def admin_user_detail(
    request: Request,
    user_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Admin user detail page"""
    if not user or not user.is_admin:
        return RedirectResponse(url="/dashboard")

    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        return RedirectResponse(url="/admin")

    from .models import UserSettings
    user_settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()

    user_clips = db.query(Clip).filter(Clip.user_id == user_id).order_by(
        Clip.detected_at.desc()
    ).limit(20).all()

    return templates.TemplateResponse("admin/user_detail.html", {
        "request": request,
        "user": user,
        "target_user": target_user,
        "settings": user_settings,
        "clips": user_clips,
    })


# Run with: uvicorn web.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
