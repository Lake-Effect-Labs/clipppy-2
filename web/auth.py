"""
Authentication Utilities

Handles JWT tokens for session management and OAuth helpers.
"""
from datetime import datetime, timedelta
from typing import Optional
import httpx
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from .config import (
    SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES,
    TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET, TWITCH_REDIRECT_URI,
    YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REDIRECT_URI,
)
from .database import get_db
from .models import User
from .encryption import encrypt_token, decrypt_token

# Security scheme
security = HTTPBearer(auto_error=False)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Get current user from JWT token (header or cookie).
    Returns None if not authenticated (for optional auth).
    """
    token = None

    # Try bearer token first
    if credentials:
        token = credentials.credentials

    # Fall back to cookie
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        return None

    payload = decode_access_token(token)
    if not payload:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    user = db.query(User).filter(User.id == int(user_id)).first()
    return user


async def get_current_user_required(
    user: Optional[User] = Depends(get_current_user)
) -> User:
    """Require authenticated user, raise 401 if not"""
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_admin_user(
    user: User = Depends(get_current_user_required)
) -> User:
    """Require admin user"""
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user


# Twitch OAuth Helpers
class TwitchOAuth:
    """Helper class for Twitch OAuth operations"""

    AUTHORIZE_URL = "https://id.twitch.tv/oauth2/authorize"
    TOKEN_URL = "https://id.twitch.tv/oauth2/token"
    VALIDATE_URL = "https://id.twitch.tv/oauth2/validate"
    USERS_URL = "https://api.twitch.tv/helix/users"

    @staticmethod
    def get_authorize_url(state: str) -> str:
        """Get Twitch OAuth authorization URL"""
        params = {
            "client_id": TWITCH_CLIENT_ID,
            "redirect_uri": TWITCH_REDIRECT_URI,
            "response_type": "code",
            "scope": "user:read:email clips:edit channel:read:stream_key",
            "state": state,
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{TwitchOAuth.AUTHORIZE_URL}?{query}"

    @staticmethod
    async def exchange_code(code: str) -> dict:
        """Exchange authorization code for tokens"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                TwitchOAuth.TOKEN_URL,
                data={
                    "client_id": TWITCH_CLIENT_ID,
                    "client_secret": TWITCH_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": TWITCH_REDIRECT_URI,
                }
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    async def refresh_token(refresh_token: str) -> dict:
        """Refresh an expired access token"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                TwitchOAuth.TOKEN_URL,
                data={
                    "client_id": TWITCH_CLIENT_ID,
                    "client_secret": TWITCH_CLIENT_SECRET,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                }
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    async def get_user_info(access_token: str) -> dict:
        """Get user info from Twitch API"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                TwitchOAuth.USERS_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Client-Id": TWITCH_CLIENT_ID,
                }
            )
            response.raise_for_status()
            data = response.json()
            if data.get("data"):
                return data["data"][0]
            return {}


# YouTube OAuth Helpers
class YouTubeOAuth:
    """Helper class for YouTube OAuth operations"""

    AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
    CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"

    SCOPES = [
        "https://www.googleapis.com/auth/youtube.upload",
        "https://www.googleapis.com/auth/youtube",
        "https://www.googleapis.com/auth/userinfo.email",
    ]

    @staticmethod
    def get_authorize_url(state: str) -> str:
        """Get YouTube OAuth authorization URL"""
        params = {
            "client_id": YOUTUBE_CLIENT_ID,
            "redirect_uri": YOUTUBE_REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(YouTubeOAuth.SCOPES),
            "state": state,
            "access_type": "offline",
            "prompt": "consent",  # Force refresh token
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{YouTubeOAuth.AUTHORIZE_URL}?{query}"

    @staticmethod
    async def exchange_code(code: str) -> dict:
        """Exchange authorization code for tokens"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                YouTubeOAuth.TOKEN_URL,
                data={
                    "client_id": YOUTUBE_CLIENT_ID,
                    "client_secret": YOUTUBE_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": YOUTUBE_REDIRECT_URI,
                }
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    async def refresh_token(refresh_token: str) -> dict:
        """Refresh an expired access token"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                YouTubeOAuth.TOKEN_URL,
                data={
                    "client_id": YOUTUBE_CLIENT_ID,
                    "client_secret": YOUTUBE_CLIENT_SECRET,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                }
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    async def get_channel_info(access_token: str) -> dict:
        """Get user's YouTube channel info"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                YouTubeOAuth.CHANNELS_URL,
                params={"part": "snippet", "mine": "true"},
                headers={"Authorization": f"Bearer {access_token}"}
            )
            response.raise_for_status()
            data = response.json()
            if data.get("items"):
                return data["items"][0]
            return {}
