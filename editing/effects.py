"""
Visual Effects Module - Shake, flash, zoom, speed ramps, and loop endings
========================================================================

Provides motion-driven effects for viral video enhancement.
"""

import numpy as np
import moviepy.editor as mp
from typing import List, Dict, Optional, Tuple, Callable
import logging
import random

logger = logging.getLogger(__name__)


def apply_micro_shake(
    clip: mp.VideoClip,
    peak_times: List[float],
    px: int = 6,
    dur: float = 0.16,
    falloff: str = 'gaussian'
) -> mp.VideoClip:
    """
    Apply micro-shake effect around audio peaks.
    
    Args:
        clip: Source video clip
        peak_times: List of peak timestamps in seconds
        px: Maximum shake displacement in pixels
        dur: Shake duration in seconds
        falloff: Falloff type ('gaussian', 'linear', 'exponential')
        
    Returns:
        Video clip with shake effect applied
    """
    if not peak_times:
        return clip
    
    def shake_position(t):
        x_offset, y_offset = 0, 0
        
        for peak_time in peak_times:
            # Time relative to this peak
            dt = abs(t - peak_time)
            
            if dt <= dur / 2:
                # Calculate falloff
                if falloff == 'gaussian':
                    intensity = np.exp(-(dt / (dur / 4)) ** 2)
                elif falloff == 'exponential':
                    intensity = np.exp(-dt / (dur / 6))
                else:  # linear
                    intensity = max(0, 1 - (dt / (dur / 2)))
                
                # Generate random shake with different frequencies for X/Y
                shake_freq_x = 25 + random.random() * 10  # 25-35 Hz
                shake_freq_y = 30 + random.random() * 10  # 30-40 Hz
                
                x_shake = intensity * px * np.sin(t * shake_freq_x * 2 * np.pi)
                y_shake = intensity * px * np.sin(t * shake_freq_y * 2 * np.pi)
                
                x_offset += x_shake
                y_offset += y_shake
        
        return ('center', int(clip.h / 2 + y_offset))
    
    shaken_clip = clip.set_position(shake_position)
    logger.info(f"Applied micro-shake to {len(peak_times)} peaks with {px}px intensity")
    
    return shaken_clip


def apply_impact_flash(
    clip: mp.VideoClip,
    peak_times: List[float],
    dur_frames: int = 2,
    fps: int = 60,
    color: str = 'white',
    opacity: float = 0.3
) -> mp.VideoClip:
    """
    Apply impact flash overlays at peak moments.
    
    Args:
        clip: Source video clip
        peak_times: List of peak timestamps in seconds
        dur_frames: Flash duration in frames
        fps: Video frame rate
        color: Flash color
        opacity: Flash opacity (0-1)
        
    Returns:
        Video clip with flash effects
    """
    if not peak_times:
        return clip
    
    dur_seconds = dur_frames / fps
    flash_clips = []
    
    for peak_time in peak_times:
        # Create white flash overlay
        flash_clip = mp.ColorClip(
            size=(clip.w, clip.h),
            color=color,
            duration=dur_seconds
        ).set_start(peak_time).set_opacity(opacity)
        
        # Add quick fade out
        flash_clip = flash_clip.fadeout(dur_seconds * 0.5)
        flash_clips.append(flash_clip)
    
    if flash_clips:
        result = mp.CompositeVideoClip([clip] + flash_clips)
        logger.info(f"Applied {len(flash_clips)} impact flashes")
        return result
    
    return clip


def apply_punch_zoom(
    clip: mp.VideoClip,
    peak_times: List[float],
    max_scale: float = 1.03,
    total_dur: float = 0.18,
    ease_type: str = 'ease_out'
) -> mp.VideoClip:
    """
    Apply punch-zoom effect at impact moments.
    
    Args:
        clip: Source video clip
        peak_times: List of peak timestamps in seconds
        max_scale: Maximum zoom scale factor
        total_dur: Total zoom duration in seconds
        ease_type: Easing function type
        
    Returns:
        Video clip with punch-zoom effect
    """
    if not peak_times:
        return clip
    
    def zoom_resize(t):
        scale = 1.0
        
        for peak_time in peak_times:
            dt = t - peak_time
            
            if 0 <= dt <= total_dur:
                # Normalized time (0 to 1) within zoom duration
                norm_t = dt / total_dur
                
                if ease_type == 'ease_out':
                    # Quick zoom in, slow zoom out
                    if norm_t <= 0.3:  # Quick zoom in (30% of duration)
                        zoom_factor = (norm_t / 0.3) * max_scale
                    else:  # Slow zoom out (70% of duration)
                        zoom_factor = max_scale * (1 - (norm_t - 0.3) / 0.7)
                else:  # linear
                    zoom_factor = max_scale * (1 - norm_t)
                
                scale = max(scale, 1.0 + zoom_factor - 1.0)
        
        return scale
    
    zoomed_clip = clip.resize(zoom_resize)
    logger.info(f"Applied punch-zoom to {len(peak_times)} peaks with {max_scale}x scale")
    
    return zoomed_clip


def speed_ramp_around(
    clip: mp.VideoClip,
    moment: Dict[str, float],
    pre_rate: float = 1.15,
    impact_rate: float = 0.6,
    chase_rate: float = 1.25,
    crossfade_dur: float = 0.05
) -> mp.VideoClip:
    """
    Apply speed ramping around a key moment.
    
    Args:
        clip: Source video clip
        moment: Dictionary with timing info from get_moment_timing()
        pre_rate: Speed multiplier for pre-impact section
        impact_rate: Speed multiplier for impact section
        chase_rate: Speed multiplier for chase section
        crossfade_dur: Crossfade duration between sections
        
    Returns:
        Video clip with speed ramping applied
    """
    if not moment:
        return clip
    
    try:
        # Split clip into sections
        pre_clip = clip.subclip(0, moment['pre_start'])
        
        pre_impact_clip = clip.subclip(
            moment['pre_start'], 
            moment['impact']
        ).fx(mp.speedx, pre_rate)
        
        impact_clip = clip.subclip(
            moment['impact'], 
            moment['post_end']
        ).fx(mp.speedx, impact_rate)
        
        post_clip = clip.subclip(moment['post_end'])
        
        # Apply chase speed to first part of post if long enough
        if post_clip.duration > 1.0:
            chase_duration = min(0.8, post_clip.duration * 0.6)
            chase_clip = post_clip.subclip(0, chase_duration).fx(mp.speedx, chase_rate)
            remaining_clip = post_clip.subclip(chase_duration)
            post_clip = mp.concatenate_videoclips([chase_clip, remaining_clip])
        
        # Concatenate with crossfades
        sections = [pre_clip, pre_impact_clip, impact_clip, post_clip]
        sections = [s for s in sections if s.duration > 0]  # Remove empty clips
        
        if len(sections) <= 1:
            return clip
        
        # Add crossfades between sections
        final_sections = [sections[0]]
        for i in range(1, len(sections)):
            if sections[i].duration > crossfade_dur * 2:
                faded_section = sections[i].crossfadein(crossfade_dur)
                final_sections.append(faded_section)
            else:
                final_sections.append(sections[i])
        
        result = mp.concatenate_videoclips(final_sections)
        
        logger.info(f"Applied speed ramp: {pre_rate}x → {impact_rate}x → {chase_rate}x")
        return result
        
    except Exception as e:
        logger.error(f"Speed ramp failed: {e}")
        return clip


def add_loop_ending(
    clip: mp.VideoClip,
    loop_frames: int = 16,
    blend_frames: int = 6,
    fps: int = 60
) -> mp.VideoClip:
    """
    Add loop-ready ending by blending first frames at the end.
    
    Args:
        clip: Source video clip
        loop_frames: Number of frames to copy from start
        blend_frames: Number of frames to crossfade
        fps: Video frame rate
        
    Returns:
        Video clip with loop ending added
    """
    if clip.duration < 1.0:  # Too short to loop
        return clip
    
    try:
        loop_duration = loop_frames / fps
        blend_duration = blend_frames / fps
        
        if loop_duration >= clip.duration * 0.5:  # Don't loop more than 50%
            loop_duration = clip.duration * 0.3
            blend_duration = loop_duration * 0.3
        
        # Extract loop segment from beginning
        loop_segment = clip.subclip(0, loop_duration)
        
        # Create blended ending
        if blend_duration > 0 and loop_duration > blend_duration:
            # Fade out end of original clip
            main_clip = clip.fadeout(blend_duration)
            
            # Fade in loop segment at the end
            loop_at_end = loop_segment.set_start(clip.duration - blend_duration).fadein(blend_duration)
            
            # Composite
            result = mp.CompositeVideoClip([main_clip, loop_at_end])
        else:
            # Simple concatenation
            result = mp.concatenate_videoclips([clip, loop_segment])
        
        logger.info(f"Added loop ending: {loop_frames} frames, {blend_frames} frame blend")
        return result
        
    except Exception as e:
        logger.error(f"Loop ending failed: {e}")
        return clip


def add_hook_overlay(
    clip: mp.VideoClip,
    hook_text: str,
    duration: float = 0.8,
    video_size: Tuple[int, int] = (1080, 1920),
    style: str = 'viral'
) -> mp.VideoClip:
    """
    Add hook text overlay at the beginning of clip.
    
    Args:
        clip: Source video clip
        hook_text: Hook text to display
        duration: Hook display duration in seconds
        video_size: Target video dimensions
        style: Hook style ('viral', 'clean', 'impact')
        
    Returns:
        Video clip with hook overlay
    """
    if not hook_text or duration <= 0:
        return clip
    
    try:
        # Style configurations
        if style == 'viral':
            font_config = {
                'fontsize': 64,
                'font': 'Arial-Bold',
                'color': 'yellow',
                'stroke_color': 'black',
                'stroke_width': 4,
                'method': 'caption'
            }
        elif style == 'impact':
            font_config = {
                'fontsize': 72,
                'font': 'Impact',
                'color': 'white',
                'stroke_color': 'red',
                'stroke_width': 3,
                'method': 'caption'
            }
        else:  # clean
            font_config = {
                'fontsize': 56,
                'font': 'Arial',
                'color': 'white',
                'stroke_color': 'black',
                'stroke_width': 2,
                'method': 'caption'
            }
        
        # Create hook text clip
        hook_clip = mp.TextClip(
            hook_text.upper(),
            size=(video_size[0] * 0.85, None),
            align='center',
            **font_config
        ).set_duration(duration)
        
        # Position in upper third
        y_pos = int(video_size[1] * 0.25)
        hook_clip = hook_clip.set_position(('center', y_pos))
        
        # Add entrance animation (scale + fade)
        def scale_in(t):
            if t <= 0.15:  # Quick scale in
                return 0.5 + (t / 0.15) * 0.5
            elif t >= duration - 0.2:  # Fade out
                fade_progress = (t - (duration - 0.2)) / 0.2
                return 1.0 - fade_progress * 0.3
            else:
                return 1.0
        
        hook_clip = hook_clip.resize(scale_in)
        
        # Add micro-zoom on first frame for motion
        if clip.duration > 0.1:
            intro_clip = clip.subclip(0, 0.1).resize(1.03)
            main_clip = clip.subclip(0.1)
            base_clip = mp.concatenate_videoclips([intro_clip, main_clip])
        else:
            base_clip = clip
        
        # Composite hook over video
        result = mp.CompositeVideoClip([base_clip, hook_clip])
        
        logger.info(f"Added hook overlay: '{hook_text}' for {duration}s")
        return result
        
    except Exception as e:
        logger.error(f"Hook overlay failed: {e}")
        return clip


def add_random_overlay(
    clip: mp.VideoClip,
    overlay_path: str,
    peak_times: List[float],
    size_ratio: float = 0.22,
    duration_range: Tuple[float, float] = (0.8, 1.2)
) -> mp.VideoClip:
    """
    Add random overlay asset at strongest peak.
    
    Args:
        clip: Source video clip
        overlay_path: Path to overlay file (PNG/WebM with alpha)
        peak_times: List of peak timestamps
        size_ratio: Overlay size as ratio of video width
        duration_range: Min/max duration for overlay
        
    Returns:
        Video clip with overlay added
    """
    if not peak_times or not overlay_path:
        return clip
    
    try:
        # Use strongest (first) peak
        peak_time = peak_times[0]
        overlay_duration = random.uniform(*duration_range)
        
        # Load overlay
        if overlay_path.endswith('.webm'):
            overlay = mp.VideoFileClip(overlay_path, has_mask=True)
        else:
            overlay = mp.ImageClip(overlay_path, transparent=True)
        
        overlay = overlay.set_duration(overlay_duration)
        
        # Resize overlay
        target_width = int(clip.w * size_ratio)
        overlay = overlay.resize(width=target_width)
        
        # Position randomly but avoid edges
        margin = 0.1
        x_range = (clip.w * margin, clip.w * (1 - margin) - overlay.w)
        y_range = (clip.h * margin, clip.h * (1 - margin) - overlay.h)
        
        x_pos = random.randint(int(x_range[0]), int(x_range[1]))
        y_pos = random.randint(int(y_range[0]), int(y_range[1]))
        
        # Set timing and position
        overlay = overlay.set_start(max(0, peak_time - 0.12))  # Slight anticipation
        overlay = overlay.set_position((x_pos, y_pos))
        
        # Add entrance/exit animation
        overlay = overlay.fadein(0.1).fadeout(0.2)
        
        # Composite
        result = mp.CompositeVideoClip([clip, overlay])
        
        logger.info(f"Added overlay at {peak_time:.2f}s: {overlay_path}")
        return result
        
    except Exception as e:
        logger.error(f"Overlay addition failed: {e}")
        return clip

