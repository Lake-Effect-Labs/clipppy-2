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
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Deque
import asyncio
import websockets
import json
import requests

logger = logging.getLogger(__name__)

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
            
            logger.info(f"🎯 Viral detector configured for {streamer_name}")
            logger.info(f"   Score threshold: {self.score_threshold} (adaptive system enabled)")
            logger.info(f"   Min unique chatters: {self.min_unique_chatters}")
            logger.info(f"   Cooldown: {self.cooldown_seconds}s")
            logger.info(f"   Daily clip target: {self.daily_clip_target}")
        else:
            # Use defaults
            self.score_threshold = self.default_score_threshold
            self.min_unique_chatters = self.default_min_unique_chatters
            self.cooldown_seconds = self.default_cooldown_seconds
            self.baseline_window_minutes = self.default_baseline_window_minutes
            logger.info(f"🎯 Viral detector using default settings for {streamer_name}")
        
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
            logger.info(f"✅ ADAPTIVE: On track ({self.clips_created_today}/{expected_clips:.1f} clips after {stream_hours:.1f}h)")
            return
        
        # Apply the new threshold with bounds checking
        self.score_threshold = max(0.1, min(2.0, new_threshold))  # Keep between 0.1 and 2.0
        
        logger.info(f"🎯 ADAPTIVE: Stream progress: {stream_progress*100:.1f}% ({stream_hours:.1f}h)")
        logger.info(f"🎯 ADAPTIVE: Current clips: {self.clips_created_today}/{self.daily_clip_target}")
        logger.info(f"🎯 ADAPTIVE: New threshold: {self.score_threshold:.3f}")
    
    def record_clip_created(self):
        """Record that a clip was created for adaptive threshold tracking"""
        self.clips_created_today += 1
        logger.info(f"📊 ADAPTIVE: Clip #{self.clips_created_today} created (target: {self.daily_clip_target})")
        
        # Check if we should adjust threshold after clip creation
        self.adjust_adaptive_threshold()
        
    def set_current_streamer(self, streamer_config: Dict):
        """Set the current streamer configuration"""
        streamer_name = streamer_config.get('name')
        self.configure_for_streamer(streamer_name)
    
    def add_chat_message(self, username: str, user_id: str, message: str, is_first_message: bool = False):
        """Add a chat message to analysis"""
        msg = ChatMessage(
            timestamp=time.time(),
            user_id=user_id,
            username=username,
            message=message,
            is_first_message=is_first_message
        )
        self.chat_messages.append(msg)
    
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
        
        if len(baseline_messages) < 10:  # Need minimum data
            return 1.0  # Default baseline
        
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
        """Calculate the viral moment score and component breakdown"""
        # Always calculate current MPS and unique chatters for monitoring
        current_mps, unique_chatters = self.calculate_messages_per_second(self.analysis_window_seconds)
        
        # Skip if stream just started (need baseline)
        if self.stream_start_time and (time.time() - self.stream_start_time) < (self.baseline_window_minutes * 60):  # Use config value
            return 0.0, {"reason": "building_baseline", "unique_chatters": unique_chatters, "current_mps": current_mps}
        
        # 1. Chat velocity component
        baseline_mps = self.calculate_baseline_mps()
        mps_zscore = self.calculate_chat_zscore(current_mps, baseline_mps)
        chat_component = self.weight_chat * min(max(mps_zscore / 2.5, 0), 1)
        
        # 2. Viewer delta component
        viewer_delta_percent = self.calculate_viewer_delta()
        viewer_component = self.weight_viewers * min(max(viewer_delta_percent / 15.0, 0), 1)
        
        # 3. Engagement events component
        engagement_score = self.calculate_engagement_score()
        engagement_component = self.weight_events * engagement_score
        
        # 4. Follow rate component
        follow_zscore = self.calculate_follow_rate_zscore()
        follow_component = self.weight_follows * min(max(follow_zscore / 3.0, 0), 1)
        
        # Total score
        total_score = chat_component + viewer_component + engagement_component + follow_component
        
        # Component breakdown for debugging
        breakdown = {
            "total_score": total_score,
            "chat_component": chat_component,
            "viewer_component": viewer_component,
            "engagement_component": engagement_component,
            "follow_component": follow_component,
            "current_mps": current_mps,
            "baseline_mps": baseline_mps,
            "mps_zscore": mps_zscore,
            "unique_chatters": unique_chatters,
            "viewer_delta_percent": viewer_delta_percent,
            "engagement_score": engagement_score,
            "follow_zscore": follow_zscore,
            "threshold": self.score_threshold
        }
        
        return total_score, breakdown
    
    def should_create_clip(self) -> Tuple[bool, str, Dict]:
        """Determine if a clip should be created based on viral score"""
        current_time = time.time()
        
        # Run adaptive threshold adjustment (checks every 30 minutes automatically)
        self.adjust_adaptive_threshold()
        
        # Check cooldown
        if current_time - self.last_clip_time < self.cooldown_seconds:
            return False, f"Cooldown active ({self.cooldown_seconds}s)", {}
        
        # Calculate viral score
        score, breakdown = self.calculate_viral_score()
        
        # Add adaptive threshold info to breakdown
        breakdown['adaptive_threshold'] = self.score_threshold
        breakdown['original_threshold'] = self.original_threshold or self.score_threshold
        breakdown['clips_today'] = self.clips_created_today
        breakdown['clip_target'] = self.daily_clip_target
        
        # Debug logging
        logger.info(f"🧪 Viral Debug - Score: {score:.3f}, Breakdown: {breakdown}")
        
        # Check unique chatters requirement
        unique_chatters = breakdown.get("unique_chatters", 0)
        logger.info(f"🧪 Chat Debug - Unique: {unique_chatters}, Min Required: {self.min_unique_chatters}")
        
        if unique_chatters < self.min_unique_chatters:
            return False, f"Not enough unique chatters ({unique_chatters}/{self.min_unique_chatters})", breakdown
        
        # Check if score meets threshold
        if score >= self.score_threshold:
            self.last_clip_time = current_time
            
            # Record clip creation for adaptive system
            self.record_clip_created()
            
            reason = f"Viral score: {score:.3f} (threshold: {self.score_threshold})"
            if breakdown.get("chat_component", 0) > 0.3:
                reason += f" - Chat spike: {breakdown.get('current_mps', 0):.1f} msgs/s"
            if breakdown.get("viewer_component", 0) > 0.2:
                reason += f" - Viewer surge: +{breakdown.get('viewer_delta_percent', 0):.1f}%"
            if breakdown.get("engagement_component", 0) > 0.1:
                reason += f" - High engagement events"
            
            return True, reason, breakdown
        
        return False, f"Score too low: {score:.3f}/{self.score_threshold}", breakdown
    
    def set_stream_status(self, is_live: bool):
        """Update stream status"""
        if is_live and not self.is_live:
            # Stream went live
            self.stream_start_time = time.time()
            self.is_live = True
            logger.info("🔴 Stream went live - starting viral detection")
            
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
