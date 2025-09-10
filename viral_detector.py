#!/usr/bin/env python3
"""
Advanced Viral Moment Detection Algorithm
=========================================

API-backed viral moment detection using Twitch's real-time signals:
- Chat velocity and z-scores
- Viewer count deltas
- High-value engagement events (raids, subs, bits, hype trains)
- Follow rate analysis

Uses EventSub + Helix polling for comprehensive signal analysis.
"""

import logging
import time
import statistics
import re
import hashlib
from math import sqrt
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Deque
import asyncio
import websockets
import json
import requests

logger = logging.getLogger(__name__)

# TikTok-native signal detection helpers
EMOTE_REGEX = re.compile(r'(:[a-zA-Z0-9_]+:|[\U0001F300-\U0001FAFF])')
ALLCAPS_REGEX = re.compile(r'^[A-Z0-9\s\!\?]{6,}$')

POS_WORDS = {"pog","poggers","insane","wow","holy","wtf","lets go","no way","clutch","ace"}
NEG_WORDS = {"throw","choke","trash","rigged","worst","mad","tilt"}

def simhash(tokens: list[str]) -> int:
    """Tiny simhash over tokens for novelty de-dupe."""
    v = [0]*64
    for t in tokens:
        h = int(hashlib.md5(t.encode('utf-8')).hexdigest(), 16)
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i, val in enumerate(v):
        if val > 0: out |= (1 << i)
    return out

def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")

def clamp(x, lo, hi): return max(lo, min(hi, x))

@dataclass
class ChatMessage:
    """Chat message data"""
    timestamp: float
    user_id: str
    username: str
    message: str
    is_first_message: bool = False

@dataclass
class EngagementEvent:
    """High-value engagement event"""
    timestamp: float
    event_type: str  # 'raid', 'gift_sub', 'bits', 'hype_train', 'follow'
    value: float     # raid_viewers, sub_count, bit_amount, etc.
    weight: float    # calculated weight for scoring

class ViralDetector:
    """Advanced viral moment detection using multiple Twitch signals"""
    
    def __init__(self, config: Dict):
        """Initialize viral detector with configuration"""
        self.config = config
        self.current_streamer = None
        
        # Algorithm configuration (defaults from global)
        alg_config = config.get('global', {}).get('viral_algorithm', {})
        self.default_score_threshold = alg_config.get('score_threshold', 1.0)
        self.default_min_unique_chatters = alg_config.get('min_unique_chatters', 30)
        self.default_cooldown_seconds = alg_config.get('cooldown_seconds', 120)
        self.default_baseline_window_minutes = alg_config.get('baseline_window_minutes', 10)
        
        # Set initial values (will be overridden by configure_for_streamer)
        self.score_threshold = self.default_score_threshold
        self.min_unique_chatters = self.default_min_unique_chatters
        self.cooldown_seconds = self.default_cooldown_seconds
        self.baseline_window_minutes = self.default_baseline_window_minutes
        self.analysis_window_seconds = alg_config.get('analysis_window_seconds', 15)
        
        # Scoring weights
        weights = alg_config.get('weights', {})
        self.weight_chat = weights.get('chat_velocity', 0.40)
        self.weight_viewers = weights.get('viewer_delta', 0.30)
        self.weight_events = weights.get('engagement_events', 0.20)
        self.weight_follows = weights.get('follow_rate', 0.10)
        
        # Data storage
        self.chat_messages: Deque[ChatMessage] = deque(maxlen=2000)  # Last ~30 min of chat
        self.viewer_history: Deque[Tuple[float, int]] = deque(maxlen=1000)  # (timestamp, viewers)
        self.engagement_events: Deque[EngagementEvent] = deque(maxlen=500)  # Last hour of events
        self.follow_events: Deque[float] = deque(maxlen=1000)  # Follow timestamps
        
        # TikTok-native signal state
        self.synergy_window_seconds = alg_config.get('synergy_window_seconds', 8)
        self.min_separation_seconds = alg_config.get('min_separation_seconds', 45)
        self.std_floor = alg_config.get('std_floor', 0.1)
        
        # profile selection
        self.profile = alg_config.get('profile', 'gaming')  # may be overwritten per-streamer
        
        # novelty detection
        self._last_chat_simhash: Optional[int] = None
        self._last_trigger_ts: float = 0.0
        
        # quality filters state
        self.first_time_chatters: Deque[float] = deque(maxlen=1000)  # timestamps of first messages
        self.sub_only_mode: bool = False  # set via EventSub if you subscribe to mode changes
        self.raid_active_until: float = 0.0
        self.brb_ad_until: float = 0.0
        
        # new signal weights (will be set by profile)
        self.weight_kw = 0.15
        self.weight_sent = 0.05
        self.weight_emote = 0.05
        
        # DEBUG COUNTER - Remove after testing
        self._debug_check_count = 0
        
        # Analysis state
        self.last_clip_time = 0
        self.baseline_mps = 0.0
        self.baseline_fpm = 0.0
        self.stream_start_time = None
        
        # Adaptive threshold system
        self.original_threshold = None  # Store original threshold
        self.clips_created_today = 0
        self.daily_clip_target = 5  # Target clips per day
        self.last_threshold_check = 0
        self.threshold_check_interval = 1800  # Check every 30 minutes (1800 seconds)
        
        # Stream status
        self.is_live = False
    
    def configure_for_streamer(self, streamer_name: str):
        """Configure algorithm settings for specific streamer"""
        self.current_streamer = streamer_name
        
        # Find streamer config
        streamer_config = None
        for streamer in self.config.get('streamers', []):
            if streamer.get('name') == streamer_name:
                streamer_config = streamer
                break
        
        if streamer_config and 'viral_algorithm' in streamer_config:
            # Use streamer-specific settings
            alg_config = streamer_config['viral_algorithm']
            self.score_threshold = alg_config.get('score_threshold', self.default_score_threshold)
            self.original_threshold = self.score_threshold  # Store original for adaptive system
            self.min_unique_chatters = alg_config.get('min_unique_chatters', self.default_min_unique_chatters)
            self.cooldown_seconds = alg_config.get('cooldown_seconds', self.default_cooldown_seconds)
            self.baseline_window_minutes = alg_config.get('baseline_window_minutes', self.default_baseline_window_minutes)
            
            logger.info(f"🎯 {streamer_name.upper()} | Threshold: {self.score_threshold} | Chatters: {self.min_unique_chatters}+ | Cooldown: {self.cooldown_seconds}s | Target: {self.daily_clip_target}/day")
        else:
            # Use defaults
            self.score_threshold = self.default_score_threshold
            self.min_unique_chatters = self.default_min_unique_chatters
            self.cooldown_seconds = self.default_cooldown_seconds
            self.baseline_window_minutes = self.default_baseline_window_minutes
            logger.info(f"🎯 Viral detector using default settings for {streamer_name}")
        
        # Load profile settings
        self.profile = streamer_config.get('profile', self.profile) if streamer_config else self.profile
        
        # apply profile weights if present
        profiles = self.config.get('profiles', {})
        if self.profile in profiles:
            p = profiles[self.profile]
            w = p.get('weights', {})
            self.weight_chat = w.get('chat_velocity', self.weight_chat)
            self.weight_viewers = w.get('viewer_delta', self.weight_viewers)
            self.weight_events = w.get('engagement_events', self.weight_events)
            self.weight_follows = w.get('follow_rate', self.weight_follows)
            # new weights
            self.weight_kw = w.get('keyword_burst', 0.15)
            self.weight_sent = w.get('sentiment_swing', 0.05)
            self.weight_emote = w.get('emote_density', 0.05)
            # thresholds
            self.score_threshold = p.get('thresholds', {}).get('score_threshold', self.score_threshold)
            logger.info(f"📱 Applied {self.profile} profile weights to {streamer_name}")
        else:
            # defaults for new weights
            self.weight_kw = 0.15
            self.weight_sent = 0.05
            self.weight_emote = 0.05
            logger.info(f"📱 Using default profile weights for {streamer_name}")
        
        # Reset stream state for new streamer
        self.stream_start_time = None
        self.last_clip_time = 0
        self.baseline_mps = 0.0
        self.baseline_fpm = 0.0
        self.chat_messages.clear()
        self.engagement_events.clear()
        self.viewer_history.clear()
        self.follow_events.clear()
    
    def adjust_adaptive_threshold(self):
        """Adjust threshold based on stream progress and clip count to guarantee daily target"""
        if not self.stream_start_time or not self.original_threshold:
            return
        
        current_time = time.time()
        
        # Only check every 30 minutes to avoid constant adjustments
        if current_time - self.last_threshold_check < self.threshold_check_interval:
            return
            
        self.last_threshold_check = current_time
        
        # Calculate stream progress (hours since start)
        stream_hours = (current_time - self.stream_start_time) / 3600
        
        # Estimate expected stream duration (assume 6-8 hour streams on average)
        estimated_stream_duration = 7.0  # hours
        stream_progress = min(stream_hours / estimated_stream_duration, 1.0)
        
        # Calculate expected clips by this point
        expected_clips = self.daily_clip_target * stream_progress
        clips_deficit = expected_clips - self.clips_created_today
        
        # Calculate threshold adjustment based on deficit
        if clips_deficit > 1.0:  # We're behind by more than 1 clip
            # Lower threshold to get more clips (more aggressive)
            adjustment_factor = max(0.5, 1.0 - (clips_deficit * 0.15))  # Lower by up to 50%
            new_threshold = self.original_threshold * adjustment_factor
            
            logger.info(f"📉 ADAPTIVE: Behind on clips ({self.clips_created_today}/{expected_clips:.1f})")
            logger.info(f"📉 ADAPTIVE: Lowering threshold {self.score_threshold:.3f} → {new_threshold:.3f}")
            
        elif clips_deficit < -0.5:  # We're ahead by more than 0.5 clips
            # Raise threshold to be more selective (less aggressive)
            adjustment_factor = min(1.5, 1.0 + (abs(clips_deficit) * 0.1))  # Raise by up to 50%
            new_threshold = self.original_threshold * adjustment_factor
            
            logger.info(f"📈 ADAPTIVE: Ahead on clips ({self.clips_created_today}/{expected_clips:.1f})")
            logger.info(f"📈 ADAPTIVE: Raising threshold {self.score_threshold:.3f} → {new_threshold:.3f}")
            
        else:
            # We're on track, no adjustment needed
            new_threshold = self.score_threshold
            # Only log if significant change or first check
            if not hasattr(self, '_last_adaptive_log') or stream_hours - self._last_adaptive_log > 1.0:
                logger.info(f"✅ On track: {self.clips_created_today}/{self.daily_clip_target} clips ({stream_hours:.1f}h)")
                self._last_adaptive_log = stream_hours
            return
        
        # Apply the new threshold with bounds checking
        self.score_threshold = max(0.1, min(2.0, new_threshold))  # Keep between 0.1 and 2.0
        
        # Removed redundant adaptive logging - already covered above
    
    def record_clip_created(self):
        """Record that a clip was created for adaptive threshold tracking"""
        self.clips_created_today += 1
        logger.info(f"📊 Clip #{self.clips_created_today}/{self.daily_clip_target} created")
        
        # Check if we should adjust threshold after clip creation
        self.adjust_adaptive_threshold()
        
    def set_current_streamer(self, streamer_config: Dict):
        """Set the current streamer configuration"""
        streamer_name = streamer_config.get('name')
        self.configure_for_streamer(streamer_name)
    
    def add_chat_message(self, username: str, user_id: str, message: str, is_first_message: bool = False):
        """Add a chat message to analysis"""
        ts = time.time()
        msg = ChatMessage(
            timestamp=ts,
            user_id=user_id,
            username=username,
            message=message,
            is_first_message=is_first_message
        )
        self.chat_messages.append(msg)
        
        # Track first-time chatters for quality metrics
        if is_first_message:
            self.first_time_chatters.append(ts)
    
    def add_viewer_data(self, viewer_count: int):
        """Add viewer count data point"""
        self.viewer_history.append((time.time(), viewer_count))
    
    def add_engagement_event(self, event_type: str, value: float):
        """Add high-value engagement event"""
        # Calculate event weight based on type and value
        weight = self._calculate_event_weight(event_type, value)
        
        event = EngagementEvent(
            timestamp=time.time(),
            event_type=event_type,
            value=value,
            weight=weight
        )
        self.engagement_events.append(event)
        logger.info(f"🔥 Engagement event: {event_type} (value: {value}, weight: {weight:.2f})")
    
    def add_follow_event(self):
        """Add a follow event"""
        self.follow_events.append(time.time())
    
    def _calculate_event_weight(self, event_type: str, value: float) -> float:
        """Calculate weight for engagement events"""
        if event_type == 'hype_train':
            return 1.0  # Hype train is always max weight
        elif event_type == 'raid':
            return min(value / 1000.0, 1.0)  # Scale by raid size
        elif event_type == 'gift_sub':
            return 0.5 if value >= 5 else 0.0  # 5+ gift subs
        elif event_type == 'bits':
            return 0.3 if value >= 500 else 0.0  # 500+ bits
        elif event_type == 'sub' or event_type == 'resub':
            return 0.1  # Individual subs
        else:
            return 0.0
    
    def calculate_messages_per_second(self, window_seconds: int = 15) -> Tuple[float, int]:
        """Calculate messages per second and unique chatters in time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Get recent messages
        recent_messages = [msg for msg in self.chat_messages if msg.timestamp > cutoff_time]
        
        # Calculate MPS
        mps = len(recent_messages) / window_seconds
        
        # Count unique chatters
        unique_user_ids = set(msg.user_id for msg in recent_messages)
        unique_chatters = len(unique_user_ids)
        
        return mps, unique_chatters
    
    def calculate_baseline_mps(self) -> float:
        """Calculate baseline messages per second over last 10 minutes"""
        current_time = time.time()
        baseline_window = self.baseline_window_minutes * 60
        cutoff_time = current_time - baseline_window
        
        # Get messages in baseline window
        baseline_messages = [msg for msg in self.chat_messages if msg.timestamp > cutoff_time]
        
        if len(baseline_messages) < 5:  # Reduced minimum data requirement
            return 0.5  # Lower default baseline to make scoring easier
        
        baseline_mps = len(baseline_messages) / baseline_window
        self.baseline_mps = baseline_mps
        return baseline_mps
    
    def calculate_viewer_delta(self) -> float:
        """Calculate viewer percentage change over last 120 seconds"""
        if len(self.viewer_history) < 2:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - 120  # 2 minutes
        
        # Get current viewers (most recent)
        current_viewers = self.viewer_history[-1][1]
        
        # Find viewers 2 minutes ago
        past_viewers = None
        for timestamp, viewers in reversed(self.viewer_history):
            if timestamp <= cutoff_time:
                past_viewers = viewers
                break
        
        if past_viewers is None or past_viewers == 0:
            return 0.0
        
        # Calculate percentage change
        delta_percent = ((current_viewers - past_viewers) / past_viewers) * 100
        return delta_percent
    
    def calculate_engagement_score(self, window_seconds: int = 60) -> float:
        """Calculate engagement event score for time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Get recent engagement events
        recent_events = [event for event in self.engagement_events if event.timestamp > cutoff_time]
        
        # Sum weights
        total_weight = sum(event.weight for event in recent_events)
        
        return min(total_weight, 1.0)  # Cap at 1.0
    
    def calculate_follow_rate_zscore(self) -> float:
        """Calculate z-score of current follow rate vs baseline"""
        current_time = time.time()
        
        # Calculate follows per minute in last 5 minutes
        recent_cutoff = current_time - 300  # 5 minutes
        recent_follows = len([ts for ts in self.follow_events if ts > recent_cutoff])
        current_fpm = recent_follows / 5.0
        
        # Calculate baseline follows per minute (last 30 minutes, excluding recent 5)
        baseline_cutoff_start = current_time - 1800  # 30 minutes
        baseline_cutoff_end = current_time - 300     # 5 minutes
        baseline_follows = [ts for ts in self.follow_events 
                          if baseline_cutoff_start < ts <= baseline_cutoff_end]
        
        if len(baseline_follows) < 5:  # Need minimum data
            return 0.0
        
        baseline_fpm = len(baseline_follows) / 25.0  # 25 minutes
        self.baseline_fpm = baseline_fpm
        
        if baseline_fpm == 0:
            return 0.0
        
        # Simple z-score approximation
        # In a real implementation, you'd calculate proper standard deviation
        z_score = (current_fpm - baseline_fpm) / max(baseline_fpm * 0.5, 0.1)
        
        return z_score
    
    def calculate_chat_zscore(self, current_mps: float, baseline_mps: float) -> float:
        """Calculate z-score for chat velocity"""
        if baseline_mps == 0:
            return 0.0
        
        # Simple z-score approximation
        # In a real implementation, you'd calculate proper standard deviation
        z_score = (current_mps - baseline_mps) / max(baseline_mps * 0.5, 0.1)
        
        return z_score
    
    def calculate_viral_score(self) -> Tuple[float, Dict]:
        """Calculate TikTok-optimized viral moment score with multi-signal analysis"""
        current_mps, unique_chatters = self.calculate_messages_per_second(self.analysis_window_seconds)

        # Skip baseline building - allow immediate clip detection
        # This prevents the system from waiting when starting monitoring on an already-live stream
        # if self.stream_start_time and (time.time() - self.stream_start_time) < 30:
        #     return 0.0, {"reason":"building_baseline","unique_chatters":unique_chatters,"current_mps":current_mps}

        baseline_mps = self.calculate_baseline_mps()
        mps_z = self.calculate_chat_zscore(current_mps, baseline_mps)

        viewer_delta_percent = self.calculate_viewer_delta()
        engagement_score = self.calculate_engagement_score()
        follow_z = self.calculate_follow_rate_zscore()

        # new TikTok-native signals
        kw_z = self.keyword_burst_z(self.analysis_window_seconds)
        sent_z = self.sentiment_swing_z(self.analysis_window_seconds)
        emote_d = self.emote_density(10)

        # normalize to 0..1 where needed
        chat_component = self.weight_chat * clamp(mps_z/2.5, 0, 1)
        viewer_component = self.weight_viewers * clamp(viewer_delta_percent/15.0, 0, 1)
        events_component = self.weight_events * clamp(engagement_score, 0, 1)
        follow_component = self.weight_follows * clamp(follow_z/3.0, 0, 1)
        kw_component = self.weight_kw * clamp(kw_z/2.5, 0, 1)
        sent_component = self.weight_sent * clamp(sent_z/2.0, 0, 1)
        emote_component = self.weight_emote * clamp((emote_d - 0.05)/0.10, 0, 1)  # boost when >5% msgs include emotes

        total = (chat_component + viewer_component + events_component +
                 follow_component + kw_component + sent_component + emote_component)

        breakdown = {
            "total_score": total,
            "components": {
                "chat": chat_component, "viewer": viewer_component, "events": events_component,
                "follow": follow_component, "keyword": kw_component, "sentiment": sent_component,
                "emote": emote_component
            },
            "z": {"chat": mps_z, "kw": kw_z, "sent": sent_z, "follow": follow_z},
            "emote_density": emote_d,
            "current_mps": current_mps,
            "baseline_mps": baseline_mps,
            "unique_chatters": unique_chatters,
            "viewer_delta_percent": viewer_delta_percent,
            "engagement_score": engagement_score,
            "threshold": self.score_threshold
        }

        # core gate (multi-signal within synergy window)
        now = time.time()
        core_ok = 0
        if mps_z >= self.config["global"]["viral_algorithm"]["core_gate"]["thresholds"]["chat_velocity_z"]: core_ok += 1
        if engagement_score >= self.config["global"]["viral_algorithm"]["core_gate"]["thresholds"]["engagement_events"]: core_ok += 1
        if kw_z >= self.config["global"]["viral_algorithm"]["core_gate"]["thresholds"]["keyword_burst_z"]: core_ok += 1
        # hype level proxy: treat big engagement burst as hype; or integrate hype_train 0..1 if you store it
        hype_level = clamp(engagement_score, 0, 1)  # replace if you track real hype_train
        if hype_level >= self.config["global"]["viral_algorithm"]["core_gate"]["thresholds"]["hype_level"]: core_ok += 1
        breakdown["core_ok"] = core_ok

        # penalties / quality gates
        penalties = 0.0
        if now < self.raid_active_until: penalties += self.config["global"]["viral_algorithm"]["penalties"]["raid_dampen"]
        if now < self.brb_ad_until: penalties += self.config["global"]["viral_algorithm"]["penalties"]["brb_or_ad"]
        if self.sub_only_mode: penalties += self.config["global"]["viral_algorithm"]["penalties"]["sub_only_chat"]
        total_after_penalty = max(0.0, total - penalties)

        breakdown["penalties"] = penalties
        breakdown["total_after_penalty"] = total_after_penalty

        # ALWAYS DEBUG LOGGING
        logger.info(f"🧪 VIRAL DEBUG | Score: {total_after_penalty:.3f}/{self.score_threshold}")
        logger.info(f"   📊 Components: chat={chat_component:.3f} viewer={viewer_component:.3f} events={events_component:.3f}")
        logger.info(f"   🎯 TikTok: kw={kw_component:.3f} sent={sent_component:.3f} emote={emote_component:.3f}")
        logger.info(f"   📈 Z-scores: chat={mps_z:.2f} kw={kw_z:.2f} sent={sent_z:.2f}")
        logger.info(f"   🚪 Core gate: {core_ok}/4 signals | penalties={penalties:.2f}")
        logger.info(f"   👥 Chatters: {unique_chatters} | MPS: {current_mps:.1f} | Baseline: {baseline_mps:.1f}")
        logger.info(f"   🎲 Gate req: {self.config['global']['viral_algorithm']['core_gate']['require_any']} | Chat msgs in last {self.analysis_window_seconds}s: {len(self._window_msgs(self.analysis_window_seconds))}")

        return total_after_penalty, breakdown
    
    def should_create_clip(self) -> Tuple[bool, str, Dict]:
        """TikTok-optimized clip creation with quality gates and novelty detection"""
        now = time.time()
        self.adjust_adaptive_threshold()
        
        # DEBUG COUNTER - Remove after testing
        self._debug_check_count += 1
        if self._debug_check_count % 5 == 0:  # Log every 5th check
            baseline_time = (now - self.stream_start_time) if self.stream_start_time else 999
            logger.info(f"🧪 VIRAL CHECK #{self._debug_check_count} | Baseline time: {baseline_time:.0f}s | Stream start: {self.stream_start_time}")

        if now - self.last_clip_time < self.cooldown_seconds:
            return False, f"Cooldown active ({self.cooldown_seconds}s)", {}

        score, br = self.calculate_viral_score()

        unique_chatters = br.get("unique_chatters", 0)
        min_chatters_needed = max(self.min_unique_chatters, self.config["global"]["viral_algorithm"]["quality"]["min_unique_chatters"])
        logger.info(f"   🔍 Quality Check: Unique chatters {unique_chatters} >= {min_chatters_needed}")
        if unique_chatters < min_chatters_needed:
            return False, f"Not enough unique chatters ({unique_chatters}/{min_chatters_needed})", br

        # first-time chatter freshness
        ftr = self.first_time_ratio(60)
        min_ftr = self.config["global"]["viral_algorithm"]["quality"]["min_first_time_ratio"]
        logger.info(f"   🔍 Quality Check: First-time ratio {ftr:.3f} >= {min_ftr}")
        if ftr < min_ftr:
            return False, f"Low first-time chatter ratio ({ftr:.3f} < {min_ftr})", br

        # viewer floor
        current_viewers = self.viewer_history[-1][1] if self.viewer_history else 0
        min_viewers = self.config["global"]["viral_algorithm"]["quality"]["min_viewer_count"]
        logger.info(f"   🔍 Quality Check: Viewers {current_viewers} >= {min_viewers}")
        if self.viewer_history and current_viewers < min_viewers:
            return False, f"Viewer count too low ({current_viewers} < {min_viewers})", br

        # synergy gate
        require_any = self.config["global"]["viral_algorithm"]["core_gate"]["require_any"]
        core_ok = br.get("core_ok", 0)
        logger.info(f"   🔍 Core Gate: {core_ok} >= {require_any} (DISABLED if require_any=0)")
        if core_ok < require_any:
            return False, f"Synergy not met ({core_ok}/{require_any})", br

        # novelty / min separation
        if now - self._last_trigger_ts < self.min_separation_seconds:
            return False, f"Min separation {self.min_separation_seconds}s", br
        sh = self.chat_window_simhash(self.analysis_window_seconds)
        if sh is not None and self._last_chat_simhash is not None and hamming(sh, self._last_chat_simhash) < 6:
            return False, "Novelty fail (similar to last clip)", br

        # final threshold
        if score >= self.score_threshold:
            self.last_clip_time = now
            self._last_trigger_ts = now
            if sh is not None: self._last_chat_simhash = sh
            self.record_clip_created()

            reason = f"Viral score {score:.2f} ≥ {self.score_threshold} | synergy ok"
            
            # DEBUG LOGGING - Remove after testing
            logger.info(f"🎬 CLIP TRIGGERED! {reason}")
            logger.info(f"   ✅ Quality: {unique_chatters} chatters | {ftr:.1%} first-time | synergy {br.get('core_ok',0)}/{require_any}")
            logger.info(f"   🎯 Novelty: separation {now - self._last_trigger_ts:.0f}s | simhash OK")
            
            return True, reason, br

        # DEBUG LOGGING - Remove after testing  
        logger.info(f"❌ No clip: {f'Score too low: {score:.2f}/{self.score_threshold}'}")
        
        return False, f"Score too low: {score:.2f}/{self.score_threshold}", br
    
    def set_stream_status(self, is_live: bool):
        """Update stream status"""
        if is_live and not self.is_live:
            # Stream went live
            self.stream_start_time = time.time()
            self.is_live = True
            logger.info("🔴 STREAM LIVE - Detection started")
            
            # Clear old data
            self.chat_messages.clear()
            self.viewer_history.clear()
            self.engagement_events.clear()
            self.follow_events.clear()
            
        elif not is_live and self.is_live:
            # Stream went offline
            self.is_live = False
            self.stream_start_time = None
            logger.info("⚫ Stream went offline - stopping viral detection")
    
    def get_debug_info(self) -> Dict:
        """Get debug information about current state"""
        score, breakdown = self.calculate_viral_score()
        
        return {
            "is_live": self.is_live,
            "stream_uptime": time.time() - self.stream_start_time if self.stream_start_time else 0,
            "chat_messages_stored": len(self.chat_messages),
            "viewer_data_points": len(self.viewer_history),
            "engagement_events_stored": len(self.engagement_events),
            "follow_events_stored": len(self.follow_events),
            "baseline_mps": self.baseline_mps,
            "baseline_fpm": self.baseline_fpm,
            "last_clip_ago": time.time() - self.last_clip_time if self.last_clip_time else float('inf'),
            "current_score": score,
            "score_breakdown": breakdown,
            "config": {
                "score_threshold": self.score_threshold,
                "min_unique_chatters": self.min_unique_chatters,
                "cooldown_seconds": self.cooldown_seconds,
                "weights": {
                    "chat": self.weight_chat,
                    "viewers": self.weight_viewers,
                    "events": self.weight_events,
                    "follows": self.weight_follows
                }
            }
        }
    
    # TikTok-native signal calculators
    def _window_msgs(self, seconds: int) -> list[ChatMessage]:
        cutoff = time.time() - seconds
        return [m for m in self.chat_messages if m.timestamp > cutoff]

    def emote_density(self, seconds: int = 10) -> float:
        msgs = self._window_msgs(seconds)
        if not msgs: return 0.0
        emote_msgs = sum(1 for m in msgs if EMOTE_REGEX.search(m.message))
        return emote_msgs / max(1, len(msgs))

    def keyword_burst_z(self, seconds: int = 15) -> float:
        msgs = self._window_msgs(seconds)
        if not msgs: return 0.0
        # compile keyword set
        kw_cfg = self.config.get('keywords', {})
        kws = set(kw_cfg.get('global', []))
        kws.update(kw_cfg.get(self.profile, []))
        hits = 0
        for m in msgs:
            s = m.message.lower()
            # cheap match: keyword or ALL CAPS or many !!!??
            if any(k in s for k in kws) or ALLCAPS_REGEX.match(m.message) or s.count('!') + s.count('?') >= 3:
                hits += 1
        rate = hits / max(1, len(msgs))
        # baseline over last 10m
        base_msgs = [m for m in self.chat_messages if m.timestamp > time.time() - self.baseline_window_minutes*60]
        if len(base_msgs) < 30: return 0.0
        base_hits = 0
        kws_l = list(kws)
        for m in base_msgs:
            s = m.message.lower()
            if any(k in s for k in kws_l) or ALLCAPS_REGEX.match(m.message) or s.count('!') + s.count('?') >= 3:
                base_hits += 1
        base_rate = base_hits / max(1, len(base_msgs))
        std = max(self.std_floor, sqrt(base_rate*(1-base_rate)+1e-6))
        return (rate - base_rate) / std

    def sentiment_swing_z(self, seconds: int = 15) -> float:
        msgs = self._window_msgs(seconds)
        if not msgs: return 0.0
        def score(s: str) -> int:
            s = s.lower()
            sc = sum(1 for w in POS_WORDS if w in s) - sum(1 for w in NEG_WORDS if w in s)
            if EMOTE_REGEX.search(s): sc += 1
            return sc
        cur = sum(score(m.message) for m in msgs) / max(1, len(msgs))
        base_msgs = [m for m in self.chat_messages if m.timestamp > time.time() - self.baseline_window_minutes*60]
        if len(base_msgs) < 30: return 0.0
        base = sum(score(m.message) for m in base_msgs) / max(1, len(base_msgs))
        std = max(self.std_floor, sqrt(abs(base)) + 0.2)
        return (cur - base) / std

    def first_time_ratio(self, seconds: int = 60) -> float:
        cutoff = time.time() - seconds
        # mark first messages
        recent = [m for m in self.chat_messages if m.timestamp > cutoff]
        if not recent: return 0.0
        return sum(1 for m in recent if m.is_first_message) / len(recent)

    def chat_window_simhash(self, seconds: int = 15) -> Optional[int]:
        msgs = self._window_msgs(seconds)
        if not msgs: return None
        # tokenization: very light
        tokens = []
        for m in msgs:
            tokens.extend(re.findall(r"[A-Za-z0-9']{3,}", m.message.lower()))
        if not tokens: return None
        return simhash(tokens)


# Example usage and testing
def test_viral_detector():
    """Test the viral detector with simulated data"""
    print("🧪 Testing Viral Detection Algorithm")
    print("=" * 50)
    
    # Test configuration
    config = {
        'viral_algorithm': {
            'score_threshold': 1.0,
            'min_unique_chatters': 10,  # Lower for testing
            'cooldown_seconds': 60,
            'baseline_window_minutes': 5,  # Shorter for testing
            'analysis_window_seconds': 15,
            'weights': {
                'chat_velocity': 0.40,
                'viewer_delta': 0.30,
                'engagement_events': 0.20,
                'follow_rate': 0.10
            }
        }
    }
    
    detector = ViralDetector(config)
    detector.set_stream_status(True)
    
    # Simulate baseline data
    print("📈 Building baseline data...")
    base_time = time.time() - 300  # 5 minutes ago
    
    # Add baseline chat messages (steady rate)
    for i in range(150):  # 0.5 msgs/sec baseline
        detector.chat_messages.append(ChatMessage(
            timestamp=base_time + i * 2,
            user_id=f"user_{i % 30}",
            username=f"user_{i % 30}",
            message=f"baseline message {i}"
        ))
    
    # Add baseline viewer data
    for i in range(10):
        detector.add_viewer_data(1000 + i * 10)  # Steady growth
        time.sleep(0.01)  # Small delay to spread timestamps
    
    # Test 1: Normal activity (should not trigger)
    print("\n🔍 Test 1: Normal activity")
    for i in range(10):
        detector.add_chat_message(f"user_{i}", f"user_{i}", "normal message")
    
    should_clip, reason, breakdown = detector.should_create_clip()
    print(f"   Result: {'🔴 CLIP' if should_clip else '✅ No clip'}")
    print(f"   Reason: {reason}")
    print(f"   Score: {breakdown.get('total_score', 0):.3f}")
    
    # Test 2: Chat spike (should trigger)
    print("\n🔍 Test 2: Chat velocity spike")
    for i in range(50):  # Sudden burst
        detector.add_chat_message(f"spike_user_{i}", f"spike_user_{i}", "POGGERS NO WAY!")
    
    should_clip, reason, breakdown = detector.should_create_clip()
    print(f"   Result: {'🔴 CLIP' if should_clip else '✅ No clip'}")
    print(f"   Reason: {reason}")
    print(f"   Score: {breakdown.get('total_score', 0):.3f}")
    
    # Test 3: Engagement event spike
    print("\n🔍 Test 3: Major engagement event")
    detector.add_engagement_event('raid', 2000)  # Big raid
    detector.add_engagement_event('gift_sub', 10)  # Gift sub bomb
    
    time.sleep(1)  # Brief delay
    should_clip, reason, breakdown = detector.should_create_clip()
    print(f"   Result: {'🔴 CLIP' if should_clip else '✅ No clip'}")
    print(f"   Reason: {reason}")
    print(f"   Score: {breakdown.get('total_score', 0):.3f}")
    
    # Show debug info
    print(f"\n📊 Debug Info:")
    debug = detector.get_debug_info()
    print(f"   Current MPS: {debug['score_breakdown'].get('current_mps', 0):.2f}")
    print(f"   Baseline MPS: {debug['baseline_mps']:.2f}")
    print(f"   Unique chatters: {debug['score_breakdown'].get('unique_chatters', 0)}")
    print(f"   Engagement score: {debug['score_breakdown'].get('engagement_score', 0):.2f}")
    
    print(f"\n✅ Viral detection algorithm test complete!")


if __name__ == "__main__":
    test_viral_detector()
