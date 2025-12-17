"""
Audio Analysis Module - RMS, transient detection, and impact peaks
================================================================

Provides audio analysis for viral moment detection and timing-based effects.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def rms(signal: np.ndarray, win_ms: int = 50, sr: int = 48000) -> Tuple[np.ndarray, int]:
    """
    Calculate RMS (Root Mean Square) energy for audio signal.
    
    Args:
        signal: Audio samples as numpy array
        win_ms: Window size in milliseconds (default: 50ms)
        sr: Sample rate in Hz (default: 48000)
        
    Returns:
        Tuple of (rms_values, hop_size_samples)
    """
    hop = int(sr * win_ms / 1000)
    win = hop
    frames = 1 + (len(signal) - win) // hop if len(signal) >= win else 0
    
    if frames <= 0:
        return np.array([]), hop
        
    out = []
    for i in range(frames):
        s = signal[i * hop:i * hop + win]
        rms_val = np.sqrt((s.astype(np.float32) ** 2).mean() + 1e-9)
        out.append(rms_val)
        
    return np.array(out), hop


def zscore(x: np.ndarray, floor: float = 1e-3) -> Tuple[np.ndarray, float, float]:
    """
    Calculate z-score normalization with numerical stability.
    
    Args:
        x: Input array
        floor: Minimum standard deviation to prevent division by zero
        
    Returns:
        Tuple of (z_scores, mean, std_dev)
    """
    mu = x.mean() if x.size else 0.0
    sd = max(x.std(), floor)
    z = (x - mu) / sd
    return z, mu, sd


def detect_impacts(
    audio_array: np.ndarray, 
    sr: int,
    rms_win_ms: int = 50,
    z_thresh: float = 2.0,
    min_gap_ms: int = 250
) -> Tuple[List[float], Tuple[np.ndarray, np.ndarray, int]]:
    """
    Detect audio impact moments using RMS energy and z-score analysis.
    
    Args:
        audio_array: Raw audio samples
        sr: Sample rate
        rms_win_ms: RMS window size in milliseconds
        z_thresh: Z-score threshold for peak detection
        min_gap_ms: Minimum gap between peaks in milliseconds
        
    Returns:
        Tuple of (peak_times_seconds, (rms_values, z_scores, hop_size))
    """
    # Calculate RMS energy
    r, hop = rms(audio_array, rms_win_ms, sr)
    
    if r.size == 0:
        return [], (r, np.array([]), hop)
    
    # Calculate z-scores
    z, mu, sd = zscore(r)
    
    # Find peaks above threshold
    peaks = np.where(z > z_thresh)[0]
    
    if peaks.size == 0:
        return [], (r, z, hop)
    
    # Thin peaks by minimum gap
    keep = [peaks[0]]
    min_gap_frames = int((min_gap_ms / 1000) / (hop / sr))
    
    for p in peaks[1:]:
        if p - keep[-1] >= min_gap_frames:
            keep.append(p)
    
    # Convert frame indices to time in seconds
    times = [int(p) * hop / sr for p in keep]
    
    logger.info(f"Detected {len(times)} impact peaks at times: {[f'{t:.2f}s' for t in times]}")
    
    return times, (r, z, hop)


def analyze_loudness_envelope(
    audio_array: np.ndarray,
    sr: int,
    win_ms: int = 100
) -> Dict[str, float]:
    """
    Analyze overall loudness characteristics of audio.
    
    Args:
        audio_array: Raw audio samples
        sr: Sample rate
        win_ms: Analysis window size in milliseconds
        
    Returns:
        Dictionary with loudness statistics
    """
    rms_vals, _ = rms(audio_array, win_ms, sr)
    
    if rms_vals.size == 0:
        return {
            'peak_rms': 0.0,
            'mean_rms': 0.0,
            'dynamic_range': 0.0,
            'peak_ratio': 0.0
        }
    
    # Convert to dB
    rms_db = 20 * np.log10(np.maximum(rms_vals, 1e-6))
    
    return {
        'peak_rms': float(np.max(rms_db)),
        'mean_rms': float(np.mean(rms_db)),
        'dynamic_range': float(np.max(rms_db) - np.min(rms_db)),
        'peak_ratio': float(np.sum(rms_vals > np.mean(rms_vals) * 1.5) / len(rms_vals))
    }


def get_moment_timing(
    peak_times: List[float],
    video_duration: float,
    pre_duration: float = 0.6,
    post_duration: float = 0.8
) -> Optional[Dict[str, float]]:
    """
    Find the strongest moment for speed ramping.
    
    Args:
        peak_times: List of peak timestamps in seconds
        video_duration: Total video duration in seconds
        pre_duration: Pre-impact duration for ramp
        post_duration: Post-impact duration for ramp
        
    Returns:
        Dictionary with moment timing or None if no suitable moment
    """
    if not peak_times:
        return None
    
    # Find peak that allows for pre/post context
    for peak_time in peak_times:
        if (peak_time >= pre_duration and 
            peak_time <= video_duration - post_duration):
            return {
                'pre_start': peak_time - pre_duration,
                'impact': peak_time,
                'post_end': peak_time + post_duration,
                'pre_duration': pre_duration,
                'post_duration': post_duration
            }
    
    # Fallback to first usable peak
    if peak_times:
        peak_time = peak_times[0]
        available_pre = min(pre_duration, peak_time)
        available_post = min(post_duration, video_duration - peak_time)
        
        if available_pre > 0.2 and available_post > 0.2:  # Minimum viable durations
            return {
                'pre_start': peak_time - available_pre,
                'impact': peak_time,
                'post_end': peak_time + available_post,
                'pre_duration': available_pre,
                'post_duration': available_post
            }
    
    return None


def detect_speech_density(
    transcript_words: List[Dict],
    window_seconds: float = 2.0
) -> float:
    """
    Calculate words per second for educational content detection.
    
    Args:
        transcript_words: List of word dictionaries with 'start' and 'end' times
        window_seconds: Time window for density calculation
        
    Returns:
        Maximum words per second in any window
    """
    if not transcript_words:
        return 0.0
    
    # Get time bounds
    start_time = transcript_words[0].get('start', 0)
    end_time = transcript_words[-1].get('end', start_time + 1)
    total_duration = end_time - start_time
    
    if total_duration <= 0:
        return 0.0
    
    # Sliding window analysis
    max_density = 0.0
    current_time = start_time
    
    while current_time + window_seconds <= end_time:
        window_end = current_time + window_seconds
        words_in_window = sum(
            1 for word in transcript_words
            if (word.get('start', 0) >= current_time and 
                word.get('end', 0) <= window_end)
        )
        density = words_in_window / window_seconds
        max_density = max(max_density, density)
        current_time += window_seconds / 4  # 25% overlap
    
    return max_density

