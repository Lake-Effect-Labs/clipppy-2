"""
Stream Listener

Connects to a Twitch chat and monitors for viral moments.
One listener instance per live user.
"""
import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Callable, Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viral_detector import ViralDetector

logger = logging.getLogger(__name__)


class StreamListener:
    """
    Listens to a Twitch stream and detects viral moments.

    Uses the existing ViralDetector algorithm but in async context.
    """

    def __init__(
        self,
        user_id: int,
        twitch_username: str,
        twitch_id: str,
        settings: dict,
        on_clip_detected: Callable,
    ):
        self.user_id = user_id
        self.twitch_username = twitch_username
        self.twitch_id = twitch_id
        self.settings = settings
        self.on_clip_detected = on_clip_detected
        self.running = False

        # Initialize viral detector with user settings
        self.detector = ViralDetector(self._build_config())
        self.detector.set_stream_status(is_live=True)

    def _build_config(self) -> dict:
        """Build config dict for ViralDetector from user settings"""
        # Merge user settings with defaults
        viral_settings = self.settings.get("viral_algorithm", {})

        return {
            "global": {
                "viral_algorithm": {
                    "score_threshold": viral_settings.get("score_threshold", 0.35),
                    "min_unique_chatters": viral_settings.get("min_unique_chatters", 25),
                    "cooldown_seconds": viral_settings.get("cooldown_seconds", 900),
                    "baseline_window_minutes": viral_settings.get("baseline_window_minutes", 5),
                    "analysis_window_seconds": 15,
                    "synergy_window_seconds": 8,
                    "min_separation_seconds": 45,
                    "std_floor": 0.1,
                    "weights": {
                        "chat_velocity": 0.30,
                        "viewer_delta": 0.20,
                        "engagement_events": 0.20,
                        "follow_rate": 0.05,
                        "keyword_burst": 0.15,
                        "sentiment_swing": 0.05,
                        "emote_density": 0.05,
                    },
                    "core_gate": {
                        "require_any": 1,
                        "thresholds": {
                            "chat_velocity_z": 2.0,
                            "reaction_burst_z": 0.2,
                        }
                    },
                    "penalties": {
                        "raid_dampen": 1.0,
                        "brb_or_ad": 1.0,
                        "sub_only_chat": 1.0,
                    },
                    "quality": {
                        "min_first_time_ratio": 0.01,
                        "min_unique_chatters": 5,
                        "min_viewer_count": 5,
                    }
                }
            },
            "streamers": [],
            "profiles": {},
        }

    async def connect_to_chat(self):
        """Connect to Twitch IRC chat"""
        import websockets

        # Twitch IRC WebSocket
        uri = "wss://irc-ws.chat.twitch.tv:443"

        async with websockets.connect(uri) as websocket:
            # Anonymous connection (no auth needed for reading)
            await websocket.send("CAP REQ :twitch.tv/tags twitch.tv/commands")
            await websocket.send("PASS oauth:placeholder")
            await websocket.send("NICK justinfan12345")  # Anonymous viewer
            await websocket.send(f"JOIN #{self.twitch_username.lower()}")

            logger.info(f"Connected to #{self.twitch_username} chat")

            async for message in websocket:
                if not self.running:
                    break

                await self.process_message(message)

    async def process_message(self, raw_message: str):
        """Process a raw IRC message"""
        # Parse IRC message
        if "PRIVMSG" not in raw_message:
            # Handle PING/PONG
            if raw_message.startswith("PING"):
                return  # Would send PONG in real implementation
            return

        try:
            # Extract username and message content
            # Format: @tags :user!user@user.tmi.twitch.tv PRIVMSG #channel :message
            parts = raw_message.split(" :", 2)
            if len(parts) < 3:
                return

            tags_part = parts[0]
            user_part = parts[1].split("!")[0]
            message_content = parts[2].strip() if len(parts) > 2 else ""

            # Extract user-id from tags
            user_id = "unknown"
            display_name = user_part
            is_first_message = False

            for tag in tags_part.split(";"):
                if tag.startswith("user-id="):
                    user_id = tag.split("=")[1]
                elif tag.startswith("display-name="):
                    display_name = tag.split("=")[1]
                elif tag.startswith("first-msg="):
                    is_first_message = tag.split("=")[1] == "1"

            # Feed to viral detector
            self.detector.add_chat_message(
                username=display_name,
                user_id=user_id,
                message=message_content,
                is_first_message=is_first_message,
            )

            # Check for viral moment
            should_clip, reason, breakdown = self.detector.should_create_clip()

            if should_clip:
                logger.info(f"VIRAL MOMENT DETECTED for {self.twitch_username}: {reason}")
                await self.on_clip_detected(
                    self.user_id,
                    {
                        "viral_score": breakdown.get("total_score", 0),
                        "reason": reason,
                        "breakdown": breakdown,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

        except Exception as e:
            logger.debug(f"Error processing message: {e}")

    async def run(self):
        """Main listener loop"""
        self.running = True
        logger.info(f"Starting listener for {self.twitch_username}")

        while self.running:
            try:
                await self.connect_to_chat()
            except Exception as e:
                logger.error(f"Chat connection error for {self.twitch_username}: {e}")
                if self.running:
                    await asyncio.sleep(5)  # Reconnect delay

        logger.info(f"Listener stopped for {self.twitch_username}")

    def stop(self):
        """Stop the listener"""
        self.running = False
        self.detector.set_stream_status(is_live=False)
