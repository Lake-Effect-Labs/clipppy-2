"""
Multi-Tenant Orchestrator

Manages stream listeners for all active users.
- Polls Twitch API to check which users are live
- Spins up/down listeners dynamically
- Passes clips to enhancement workers
"""
import asyncio
import logging
import time
import os
import sys
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, field

import httpx
from sqlalchemy.orm import Session

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.database import SessionLocal
from web.models import User, Clip, UserSettings, ClipStatus, StreamSession
from web.encryption import decrypt_token
from web.config import TWITCH_CLIENT_ID, TWITCH_CLIENT_SECRET

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ListenerState:
    """State of an active listener"""
    user_id: int
    twitch_username: str
    twitch_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    clips_this_session: int = 0
    last_viral_score: float = 0.0
    task: Optional[asyncio.Task] = None


class MultiTenantOrchestrator:
    """
    Orchestrates stream listeners for all active users.

    Responsibilities:
    - Poll Twitch API to detect which users are live
    - Start/stop listener tasks for live users
    - Track listener state and statistics
    """

    def __init__(self):
        self.active_listeners: Dict[int, ListenerState] = {}  # user_id -> ListenerState
        self.app_access_token: Optional[str] = None
        self.token_expires_at: float = 0
        self.check_interval = 60  # seconds between live checks
        self.running = False

    async def get_app_access_token(self) -> str:
        """Get or refresh Twitch app access token"""
        if self.app_access_token and time.time() < self.token_expires_at - 300:
            return self.app_access_token

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://id.twitch.tv/oauth2/token",
                data={
                    "client_id": TWITCH_CLIENT_ID,
                    "client_secret": TWITCH_CLIENT_SECRET,
                    "grant_type": "client_credentials",
                }
            )
            response.raise_for_status()
            data = response.json()

            self.app_access_token = data["access_token"]
            self.token_expires_at = time.time() + data.get("expires_in", 3600)

            logger.info("Refreshed Twitch app access token")
            return self.app_access_token

    async def get_live_users(self, user_ids: list[str]) -> set[str]:
        """
        Check which Twitch users are currently live.
        Uses batch API calls (up to 100 per request).
        """
        if not user_ids:
            return set()

        live_ids = set()
        token = await self.get_app_access_token()

        async with httpx.AsyncClient() as client:
            # Batch into chunks of 100
            for i in range(0, len(user_ids), 100):
                chunk = user_ids[i:i+100]
                params = "&".join([f"user_id={uid}" for uid in chunk])

                try:
                    response = await client.get(
                        f"https://api.twitch.tv/helix/streams?{params}",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Client-Id": TWITCH_CLIENT_ID,
                        }
                    )
                    response.raise_for_status()
                    data = response.json()

                    for stream in data.get("data", []):
                        live_ids.add(stream["user_id"])

                except Exception as e:
                    logger.error(f"Error checking live status: {e}")

        return live_ids

    async def start_listener(self, user: User, settings: UserSettings):
        """Start a listener for a user"""
        logger.info(f"Starting listener for {user.twitch_username}")

        state = ListenerState(
            user_id=user.id,
            twitch_username=user.twitch_username,
            twitch_id=user.twitch_id,
        )

        # Create the listener task
        from .listener import StreamListener
        listener = StreamListener(
            user_id=user.id,
            twitch_username=user.twitch_username,
            twitch_id=user.twitch_id,
            settings=settings.to_viral_config() if settings else {},
            on_clip_detected=self.on_clip_detected,
        )

        state.task = asyncio.create_task(listener.run())
        self.active_listeners[user.id] = state

        # Create stream session record
        db = SessionLocal()
        try:
            session = StreamSession(
                user_id=user.id,
                stream_title="",  # Will be updated by listener
            )
            db.add(session)
            db.commit()
        finally:
            db.close()

    async def stop_listener(self, user_id: int):
        """Stop a listener for a user"""
        if user_id not in self.active_listeners:
            return

        state = self.active_listeners.pop(user_id)
        logger.info(f"Stopping listener for {state.twitch_username} (ran for {datetime.utcnow() - state.started_at})")

        if state.task:
            state.task.cancel()
            try:
                await state.task
            except asyncio.CancelledError:
                pass

        # Update stream session record
        db = SessionLocal()
        try:
            session = db.query(StreamSession).filter(
                StreamSession.user_id == user_id,
                StreamSession.ended_at == None
            ).order_by(StreamSession.started_at.desc()).first()

            if session:
                session.ended_at = datetime.utcnow()
                session.duration_seconds = int((session.ended_at - session.started_at).total_seconds())
                session.clips_created = state.clips_this_session
                db.commit()
        finally:
            db.close()

    async def on_clip_detected(self, user_id: int, clip_data: dict):
        """Called when a listener detects a viral moment"""
        logger.info(f"Clip detected for user {user_id}: score={clip_data.get('viral_score', 0):.3f}")

        if user_id in self.active_listeners:
            self.active_listeners[user_id].clips_this_session += 1
            self.active_listeners[user_id].last_viral_score = clip_data.get("viral_score", 0)

        # Queue clip creation and enhancement
        try:
            from celery_tasks import create_and_enhance_clip
            create_and_enhance_clip.delay(user_id, clip_data)
        except ImportError:
            # Celery not available, create clip synchronously
            logger.warning("Celery not available, processing clip synchronously")
            await self.create_clip_sync(user_id, clip_data)

    async def create_clip_sync(self, user_id: int, clip_data: dict):
        """Synchronous clip creation (fallback if Celery unavailable)"""
        import uuid
        db = SessionLocal()
        try:
            clip = Clip(
                user_id=user_id,
                clip_id=str(uuid.uuid4())[:8],
                status=ClipStatus.PENDING_REVIEW,  # Skip processing for now
                viral_score=clip_data.get("viral_score"),
                trigger_reason=clip_data.get("reason"),
                detection_breakdown=clip_data.get("breakdown"),
                suggested_title=f"Viral moment (score: {clip_data.get('viral_score', 0):.2f})",
                detected_at=datetime.utcnow(),
            )
            db.add(clip)
            db.commit()
            logger.info(f"Created clip {clip.clip_id} for user {user_id}")
        finally:
            db.close()

    async def reconcile(self):
        """Main reconciliation loop - start/stop listeners as needed"""
        db = SessionLocal()
        try:
            # Get all active users with valid tokens
            users = db.query(User).filter(
                User.is_active == True,
                User.twitch_token_encrypted != None,
            ).all()

            if not users:
                logger.debug("No active users with tokens")
                return

            # Get their Twitch IDs
            user_map = {u.twitch_id: u for u in users}
            twitch_ids = list(user_map.keys())

            # Check who's live
            live_ids = await self.get_live_users(twitch_ids)
            logger.info(f"Live check: {len(live_ids)}/{len(twitch_ids)} users live")

            # Start listeners for newly live users
            for twitch_id in live_ids:
                user = user_map[twitch_id]
                if user.id not in self.active_listeners:
                    settings = db.query(UserSettings).filter(
                        UserSettings.user_id == user.id
                    ).first()
                    await self.start_listener(user, settings)

            # Stop listeners for users who went offline
            active_user_ids = list(self.active_listeners.keys())
            for user_id in active_user_ids:
                state = self.active_listeners[user_id]
                if state.twitch_id not in live_ids:
                    await self.stop_listener(user_id)

        finally:
            db.close()

    async def run(self):
        """Main orchestrator loop"""
        logger.info("Starting Multi-Tenant Orchestrator")
        self.running = True

        while self.running:
            try:
                await self.reconcile()
            except Exception as e:
                logger.error(f"Reconciliation error: {e}")
                import traceback
                traceback.print_exc()

            await asyncio.sleep(self.check_interval)

        logger.info("Orchestrator stopped")

    def stop(self):
        """Stop the orchestrator"""
        self.running = False

    def get_status(self) -> dict:
        """Get current orchestrator status"""
        return {
            "running": self.running,
            "active_listeners": len(self.active_listeners),
            "listeners": [
                {
                    "user_id": state.user_id,
                    "twitch_username": state.twitch_username,
                    "started_at": state.started_at.isoformat(),
                    "clips_this_session": state.clips_this_session,
                    "last_viral_score": state.last_viral_score,
                }
                for state in self.active_listeners.values()
            ]
        }


async def main():
    """Run the orchestrator"""
    orchestrator = MultiTenantOrchestrator()

    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        orchestrator.stop()

        # Stop all listeners
        for user_id in list(orchestrator.active_listeners.keys()):
            await orchestrator.stop_listener(user_id)


if __name__ == "__main__":
    asyncio.run(main())
