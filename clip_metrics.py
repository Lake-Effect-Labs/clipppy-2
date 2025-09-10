#!/usr/bin/env python3
"""
Viral Clip Metrics Tracking
============================

Tracks input signals and post-publish performance for viral algorithm optimization.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ClipMetrics:
    """Track comprehensive metrics for a single clip"""
    clip_id: str
    streamer_name: str
    created_at: float
    
    # Input signals at creation time
    input_signals: Dict = None
    
    # Decision factors
    decision_flags: Dict = None
    
    # Post-publish performance (to be updated later)
    views_1h: Optional[int] = None
    views_6h: Optional[int] = None
    views_24h: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    completion_rate: Optional[float] = None
    
    # Performance classification
    is_hit: Optional[bool] = None  # True if views > threshold

class ClipMetricsTracker:
    """Track and analyze clip performance metrics"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.metrics_file = self.data_dir / "clip_metrics.json"
        self.metrics: List[ClipMetrics] = []
        self.load_metrics()
    
    def load_metrics(self):
        """Load existing metrics from JSON file"""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = [ClipMetrics(**item) for item in data]
                logger.info(f"📊 Loaded {len(self.metrics)} clip metrics")
            except Exception as e:
                logger.warning(f"⚠️ Could not load metrics: {e}")
                self.metrics = []
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        try:
            data = [asdict(metric) for metric in self.metrics]
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"💾 Saved {len(self.metrics)} clip metrics")
        except Exception as e:
            logger.error(f"❌ Could not save metrics: {e}")
    
    def record_clip_created(self, clip_id: str, streamer_name: str, 
                          input_signals: Dict, decision_flags: Dict):
        """Record metrics when a clip is created"""
        metric = ClipMetrics(
            clip_id=clip_id,
            streamer_name=streamer_name,
            created_at=time.time(),
            input_signals=input_signals,
            decision_flags=decision_flags
        )
        
        self.metrics.append(metric)
        self.save_metrics()
        
        logger.info(f"📊 Recorded metrics for clip: {clip_id}")
    
    def update_performance(self, clip_id: str, views_1h: int = None, 
                          views_6h: int = None, views_24h: int = None,
                          likes: int = None, comments: int = None, 
                          shares: int = None, completion_rate: float = None):
        """Update post-publish performance metrics"""
        for metric in self.metrics:
            if metric.clip_id == clip_id:
                if views_1h is not None: metric.views_1h = views_1h
                if views_6h is not None: metric.views_6h = views_6h
                if views_24h is not None: metric.views_24h = views_24h
                if likes is not None: metric.likes = likes
                if comments is not None: metric.comments = comments
                if shares is not None: metric.shares = shares
                if completion_rate is not None: metric.completion_rate = completion_rate
                
                # Classify as hit if 24h views > threshold
                if views_24h is not None:
                    metric.is_hit = views_24h > 50000  # 50k views threshold
                
                self.save_metrics()
                logger.info(f"📊 Updated performance for clip: {clip_id}")
                return
        
        logger.warning(f"⚠️ Clip not found for update: {clip_id}")
    
    def analyze_top_signals(self, days: int = 7) -> Dict:
        """Analyze which signals correlate with hits"""
        cutoff = time.time() - (days * 24 * 60 * 60)
        recent_metrics = [m for m in self.metrics if m.created_at > cutoff and m.is_hit is not None]
        
        if len(recent_metrics) < 10:
            return {"error": "Not enough data for analysis"}
        
        hits = [m for m in recent_metrics if m.is_hit]
        misses = [m for m in recent_metrics if not m.is_hit]
        
        analysis = {
            "period_days": days,
            "total_clips": len(recent_metrics),
            "hits": len(hits),
            "hit_rate": len(hits) / len(recent_metrics),
            "top_signals_for_hits": {},
            "suggested_weight_changes": {}
        }
        
        # Analyze signal averages for hits vs misses
        if hits and misses:
            signal_keys = ['chat', 'viewer', 'events', 'keyword', 'sentiment', 'emote']
            
            for key in signal_keys:
                hit_avg = sum(m.input_signals.get('components', {}).get(key, 0) for m in hits) / len(hits)
                miss_avg = sum(m.input_signals.get('components', {}).get(key, 0) for m in misses) / len(misses)
                
                analysis["top_signals_for_hits"][key] = {
                    "hit_avg": hit_avg,
                    "miss_avg": miss_avg,
                    "difference": hit_avg - miss_avg
                }
                
                # Suggest weight changes (simple heuristic)
                if hit_avg > miss_avg * 1.2:
                    analysis["suggested_weight_changes"][key] = "increase by 10-20%"
                elif hit_avg < miss_avg * 0.8:
                    analysis["suggested_weight_changes"][key] = "decrease by 10-20%"
        
        return analysis
    
    def get_streamer_stats(self, streamer_name: str, days: int = 30) -> Dict:
        """Get performance stats for a specific streamer"""
        cutoff = time.time() - (days * 24 * 60 * 60)
        streamer_metrics = [m for m in self.metrics 
                          if m.streamer_name == streamer_name and m.created_at > cutoff]
        
        if not streamer_metrics:
            return {"error": f"No data for {streamer_name}"}
        
        hits = [m for m in streamer_metrics if m.is_hit]
        
        return {
            "streamer": streamer_name,
            "period_days": days,
            "total_clips": len(streamer_metrics),
            "hits": len(hits),
            "hit_rate": len(hits) / len(streamer_metrics) if streamer_metrics else 0,
            "avg_views_24h": sum(m.views_24h for m in streamer_metrics if m.views_24h) / len([m for m in streamer_metrics if m.views_24h]) if [m for m in streamer_metrics if m.views_24h] else 0
        }

def test_metrics_tracker():
    """Test the metrics tracking system"""
    print("🧪 Testing Clip Metrics Tracker")
    print("=" * 50)
    
    tracker = ClipMetricsTracker()
    
    # Simulate recording a clip
    test_signals = {
        "components": {
            "chat": 0.8, "viewer": 0.6, "events": 0.4,
            "keyword": 0.9, "sentiment": 0.3, "emote": 0.7
        },
        "z": {"chat": 2.1, "kw": 2.3, "sent": 0.8}
    }
    
    test_flags = {
        "core_ok": 3,
        "final_score": 1.45,
        "profile": "gaming",
        "threshold": 1.35
    }
    
    tracker.record_clip_created("test_clip_001", "jynxzi", test_signals, test_flags)
    
    # Simulate updating performance
    tracker.update_performance("test_clip_001", views_24h=75000, likes=1200, completion_rate=0.68)
    
    # Analyze performance
    analysis = tracker.analyze_top_signals(7)
    print("📊 Analysis results:", analysis)
    
    stats = tracker.get_streamer_stats("jynxzi", 30)
    print("📈 Jynxzi stats:", stats)

if __name__ == "__main__":
    test_metrics_tracker()
