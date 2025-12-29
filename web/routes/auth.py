"""
Authentication Routes

Handles Twitch OAuth login/logout and YouTube connection.
"""
import secrets
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User, UserSettings
from ..schemas import Token, UserResponse
from ..auth import (
    TwitchOAuth, YouTubeOAuth,
    create_access_token, get_current_user, get_current_user_required
)
from ..encryption import encrypt_token
from ..config import APP_URL, ADMIN_EMAILS

router = APIRouter(prefix="/auth", tags=["auth"])

# Store OAuth states temporarily (use Redis in production)
oauth_states = {}


@router.get("/twitch")
async def twitch_login(request: Request):
    """Initiate Twitch OAuth login"""
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {"type": "twitch", "created": datetime.utcnow()}

    auth_url = TwitchOAuth.get_authorize_url(state)
    return RedirectResponse(url=auth_url)


@router.get("/twitch/callback")
async def twitch_callback(
    code: str,
    state: str,
    response: Response,
    db: Session = Depends(get_db)
):
    """Handle Twitch OAuth callback"""
    # Validate state
    if state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")
    del oauth_states[state]

    try:
        # Exchange code for tokens
        tokens = await TwitchOAuth.exchange_code(code)
        access_token = tokens["access_token"]
        refresh_token = tokens.get("refresh_token")

        # Get user info from Twitch
        user_info = await TwitchOAuth.get_user_info(access_token)
        if not user_info:
            raise HTTPException(status_code=400, detail="Failed to get user info from Twitch")

        twitch_id = user_info["id"]
        twitch_username = user_info["login"]
        twitch_display_name = user_info.get("display_name", twitch_username)
        twitch_email = user_info.get("email")
        twitch_profile_image = user_info.get("profile_image_url")

        # Find or create user
        user = db.query(User).filter(User.twitch_id == twitch_id).first()

        if not user:
            # Create new user
            user = User(
                twitch_id=twitch_id,
                twitch_username=twitch_username,
                twitch_display_name=twitch_display_name,
                twitch_email=twitch_email,
                twitch_profile_image=twitch_profile_image,
            )
            db.add(user)
            db.flush()  # Get the user ID

            # Create default settings
            settings = UserSettings(
                user_id=user.id,
                watermark_text=f"@{twitch_username}_clippy"
            )
            db.add(settings)

            # Check if admin
            if twitch_email and twitch_email in ADMIN_EMAILS:
                user.is_admin = True

        else:
            # Update existing user
            user.twitch_username = twitch_username
            user.twitch_display_name = twitch_display_name
            user.twitch_email = twitch_email
            user.twitch_profile_image = twitch_profile_image

        # Store encrypted tokens
        token_data = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": tokens.get("expires_in", 3600),
            "obtained_at": datetime.utcnow().isoformat(),
        }
        user.twitch_token_encrypted = encrypt_token(token_data)
        user.last_login_at = datetime.utcnow()

        db.commit()

        # Create session token
        session_token = create_access_token(data={"sub": str(user.id)})

        # Redirect to dashboard with cookie
        redirect_response = RedirectResponse(url="/dashboard", status_code=303)
        redirect_response.set_cookie(
            key="access_token",
            value=session_token,
            httponly=True,
            max_age=60 * 60 * 24 * 7,  # 1 week
            samesite="lax"
        )
        return redirect_response

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OAuth error: {str(e)}")


@router.get("/youtube")
async def youtube_connect(
    request: Request,
    user: User = Depends(get_current_user_required)
):
    """Initiate YouTube OAuth connection"""
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {
        "type": "youtube",
        "user_id": user.id,
        "created": datetime.utcnow()
    }

    auth_url = YouTubeOAuth.get_authorize_url(state)
    return RedirectResponse(url=auth_url)


@router.get("/youtube/callback")
async def youtube_callback(
    code: str,
    state: str,
    db: Session = Depends(get_db)
):
    """Handle YouTube OAuth callback"""
    # Validate state
    if state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    state_data = oauth_states.pop(state)
    if state_data["type"] != "youtube":
        raise HTTPException(status_code=400, detail="Invalid OAuth state type")

    user_id = state_data["user_id"]
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        # Exchange code for tokens
        tokens = await YouTubeOAuth.exchange_code(code)
        access_token = tokens["access_token"]
        refresh_token = tokens.get("refresh_token")

        # Get channel info
        channel_info = await YouTubeOAuth.get_channel_info(access_token)

        # Store encrypted tokens
        token_data = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": tokens.get("expires_in", 3600),
            "obtained_at": datetime.utcnow().isoformat(),
        }
        user.youtube_token_encrypted = encrypt_token(token_data)
        user.youtube_connected = True

        if channel_info:
            user.youtube_channel_id = channel_info.get("id")
            snippet = channel_info.get("snippet", {})
            user.youtube_channel_name = snippet.get("title")

        db.commit()

        # Redirect back to settings
        return RedirectResponse(url="/settings?youtube=connected", status_code=303)

    except Exception as e:
        return RedirectResponse(url=f"/settings?youtube=error&message={str(e)}", status_code=303)


@router.post("/youtube/disconnect")
async def youtube_disconnect(
    user: User = Depends(get_current_user_required),
    db: Session = Depends(get_db)
):
    """Disconnect YouTube account"""
    user.youtube_token_encrypted = None
    user.youtube_connected = False
    user.youtube_channel_id = None
    user.youtube_channel_name = None
    db.commit()
    return {"message": "YouTube disconnected"}


@router.get("/logout")
async def logout(response: Response):
    """Log out current user"""
    redirect_response = RedirectResponse(url="/", status_code=303)
    redirect_response.delete_cookie("access_token")
    return redirect_response


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user_required)):
    """Get current user info"""
    return user
