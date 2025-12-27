"""
Weekly Compilation Video Generator for Clipppy

Creates long-form YouTube highlight videos from raw clips.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import click

# Try importing video processing libraries
try:
    from moviepy.editor import (
        VideoFileClip, CompositeVideoClip, TextClip, 
        concatenate_videoclips, AudioFileClip
    )
    import moviepy.config as mp_config
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WeeklyCompilationCreator:
    """Creates weekly compilation videos from raw clips"""
    
    def __init__(self, clips_base_dir: str = "clips"):
        self.clips_base_dir = Path(clips_base_dir)
        
        if not MOVIEPY_AVAILABLE:
            raise ImportError(
                "MoviePy is required for video compilation. "
                "Install with: pip install moviepy"
            )
    
    def get_raw_clips_for_streamer(
        self, 
        streamer_name: str,
        days_back: int = 7,
        min_clips: Optional[int] = None,
        max_clips: Optional[int] = None,
        use_enhanced: bool = False
    ) -> List[Tuple[Path, datetime]]:
        """
        Get clips for a streamer within a time range.
        
        Args:
            streamer_name: Name of the streamer
            days_back: Number of days to look back
            min_clips: Minimum number of clips to include (will go back further if needed)
            max_clips: Maximum number of clips to include
            use_enhanced: If True, use enhanced clips instead of raw clips
            
        Returns:
            List of (clip_path, creation_time) tuples, sorted by creation time
        """
        if use_enhanced:
            clips_dir = self.clips_base_dir / streamer_name
            clip_pattern = "enhanced_*.mp4"
        else:
            clips_dir = self.clips_base_dir / streamer_name / "raw"
            clip_pattern = "raw_*.mp4"
        
        if not clips_dir.exists():
            logger.warning(f"⚠️ No clips directory found for {streamer_name} at {clips_dir}")
            return []
        
        # Get all clips
        all_clips = list(clips_dir.glob(clip_pattern))
        
        if not all_clips:
            logger.warning(f"⚠️ No clips found for {streamer_name}")
            return []
        
        # Extract timestamps and filter by date
        cutoff_time = datetime.now() - timedelta(days=days_back)
        clips_with_times = []
        
        for clip_path in all_clips:
            # Extract timestamp from filename: raw_streamername_1234567890.mp4 or enhanced_streamername_1234567890.mp4
            try:
                timestamp_str = clip_path.stem.split('_')[-1]
                timestamp = int(timestamp_str)
                clip_time = datetime.fromtimestamp(timestamp)
                
                if clip_time >= cutoff_time or min_clips:
                    clips_with_times.append((clip_path, clip_time))
            except (ValueError, IndexError) as e:
                logger.warning(f"⚠️ Could not parse timestamp from {clip_path.name}: {e}")
                continue
        
        # Sort by creation time (oldest first)
        clips_with_times.sort(key=lambda x: x[1])
        
        # Apply min/max clip constraints
        if min_clips and len(clips_with_times) < min_clips:
            logger.info(f"📊 Found {len(clips_with_times)} clips (less than min {min_clips}), including all available")
        elif min_clips and len(clips_with_times) > min_clips:
            clips_with_times = clips_with_times[-min_clips:]  # Take most recent
            
        if max_clips and len(clips_with_times) > max_clips:
            clips_with_times = clips_with_times[-max_clips:]  # Take most recent
            logger.info(f"📊 Limited to {max_clips} most recent clips")
        
        return clips_with_times
    
    def create_intro_clip(self, streamer_name: str, duration: float = 3.0) -> Optional[VideoFileClip]:
        """
        Create a simple intro clip with text.
        
        Args:
            streamer_name: Name of the streamer
            duration: Duration of intro in seconds
            
        Returns:
            VideoFileClip with intro, or None if creation failed
        """
        try:
            # Create a simple text intro
            intro_text = TextClip(
                f"{streamer_name.upper()}\nWEEKLY HIGHLIGHTS",
                fontsize=70,
                color='white',
                bg_color='black',
                size=(1920, 1080),
                method='caption',
                align='center'
            ).set_duration(duration)
            
            return intro_text
        except Exception as e:
            logger.warning(f"⚠️ Could not create intro: {e}")
            return None
    
    def create_transition(self, clip_number: int, duration: float = 1.0) -> Optional[VideoFileClip]:
        """
        Create a simple transition clip between highlights.
        
        Args:
            clip_number: The clip number being transitioned to
            duration: Duration of transition in seconds
            
        Returns:
            VideoFileClip with transition, or None if creation failed
        """
        try:
            transition_text = TextClip(
                f"HIGHLIGHT #{clip_number}",
                fontsize=50,
                color='white',
                bg_color='black',
                size=(1920, 1080),
                method='caption',
                align='center'
            ).set_duration(duration)
            
            return transition_text
        except Exception as e:
            logger.warning(f"⚠️ Could not create transition: {e}")
            return None
    
    def create_compilation(
        self,
        streamer_name: str,
        days_back: int = 7,
        output_path: Optional[str] = None,
        add_intro: bool = True,
        add_transitions: bool = True,
        max_clips: Optional[int] = None,
        min_clips: Optional[int] = None,
        use_enhanced: bool = False
    ) -> Optional[str]:
        """
        Create a weekly compilation video from clips.
        
        Args:
            streamer_name: Name of the streamer
            days_back: Number of days to look back for clips
            output_path: Optional custom output path
            add_intro: Whether to add an intro clip
            add_transitions: Whether to add transitions between clips
            max_clips: Maximum number of clips to include
            min_clips: Minimum number of clips to include
            use_enhanced: If True, use enhanced clips instead of raw clips
            
        Returns:
            Path to the output video, or None if failed
        """
        logger.info(f"🎬 Creating weekly compilation for {streamer_name}")
        logger.info(f"📅 Looking back {days_back} days")
        
        # Get clips
        clips_data = self.get_raw_clips_for_streamer(
            streamer_name, 
            days_back=days_back,
            min_clips=min_clips,
            max_clips=max_clips,
            use_enhanced=use_enhanced
        )
        
        if not clips_data:
            logger.error(f"❌ No clips found for {streamer_name}")
            return None
        
        logger.info(f"📊 Found {len(clips_data)} clips to compile")
        
        # Build video sequence
        video_clips = []
        
        try:
            # Add intro if requested
            if add_intro:
                logger.info("🎬 Creating intro...")
                intro = self.create_intro_clip(streamer_name)
                if intro:
                    video_clips.append(intro)
            
            # Add each clip with optional transitions
            # Uses RAW unenhanced clips (already 16:9 landscape from Twitch)
            for idx, (clip_path, clip_time) in enumerate(clips_data, 1):
                logger.info(f"📹 Loading clip {idx}/{len(clips_data)}: {clip_path.name}")
                
                # Add transition if requested (except before first clip)
                if add_transitions and idx > 1:
                    transition = self.create_transition(idx)
                    if transition:
                        video_clips.append(transition)
                
                # Load the raw clip (no conversion needed - already landscape)
                try:
                    clip = VideoFileClip(str(clip_path))
                    video_clips.append(clip)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {clip_path.name}: {e}")
                    continue
            
            if not video_clips:
                logger.error("❌ No video clips could be loaded")
                return None
            
            # Concatenate all clips
            logger.info(f"🔗 Concatenating {len(video_clips)} segments...")
            final_video = concatenate_videoclips(video_clips, method="compose")
            
            # Setup output path
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d")
                compilations_dir = self.clips_base_dir / streamer_name / "compilations"
                compilations_dir.mkdir(parents=True, exist_ok=True)
                output_path = compilations_dir / f"{streamer_name}_weekly_{timestamp}.mp4"
            else:
                output_path = Path(output_path)
            
            # Render final video
            logger.info(f"🎥 Rendering compilation to: {output_path}")
            logger.info(f"⏱️ Total duration: {final_video.duration:.1f} seconds ({final_video.duration/60:.1f} minutes)")
            
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                fps=30,
                preset='medium',
                threads=4
            )
            
            # Clean up
            final_video.close()
            for clip in video_clips:
                clip.close()
            
            logger.info(f"✅ Compilation complete: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to create compilation: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up on error
            for clip in video_clips:
                try:
                    clip.close()
                except:
                    pass
            
            return None


@click.command()
@click.option('--streamer', default='theburntpeanut', help='Streamer name')
@click.option('--days', default=7, help='Days to look back for clips')
@click.option('--max-clips', type=int, help='Maximum number of clips to include')
@click.option('--min-clips', type=int, help='Minimum number of clips to include')
@click.option('--no-intro', is_flag=True, help='Skip intro')
@click.option('--no-transitions', is_flag=True, help='Skip transitions')
@click.option('--output', help='Custom output path')
def create_weekly_compilation(streamer, days, max_clips, min_clips, no_intro, no_transitions, output):
    """Create a weekly compilation video from raw clips"""
    
    if not MOVIEPY_AVAILABLE:
        click.echo("❌ MoviePy is required. Install with: pip install moviepy")
        return
    
    click.echo("=" * 60)
    click.echo("🎬 CLIPPPY WEEKLY COMPILATION CREATOR")
    click.echo("=" * 60)
    click.echo()
    
    creator = WeeklyCompilationCreator()
    
    # Preview available clips
    clips_data = creator.get_raw_clips_for_streamer(
        streamer,
        days_back=days,
        min_clips=min_clips,
        max_clips=max_clips
    )
    
    if not clips_data:
        click.echo(f"❌ No raw clips found for {streamer}")
        click.echo(f"   Make sure clips exist in: clips/{streamer}/raw/")
        return
    
    click.echo(f"📊 Found {len(clips_data)} raw clips:")
    total_duration = 0
    
    for idx, (clip_path, clip_time) in enumerate(clips_data, 1):
        try:
            clip = VideoFileClip(str(clip_path))
            duration = clip.duration
            clip.close()
            total_duration += duration
            click.echo(f"   {idx}. {clip_path.name} - {duration:.1f}s - {clip_time.strftime('%Y-%m-%d %H:%M')}")
        except:
            click.echo(f"   {idx}. {clip_path.name} - (could not read duration)")
    
    click.echo()
    click.echo(f"⏱️ Estimated compilation length: {total_duration/60:.1f} minutes")
    click.echo()
    
    if not click.confirm("📹 Create compilation from these clips?", default=True):
        click.echo("❌ Cancelled")
        return
    
    # Create compilation
    result = creator.create_compilation(
        streamer_name=streamer,
        days_back=days,
        output_path=output,
        add_intro=not no_intro,
        add_transitions=not no_transitions,
        max_clips=max_clips,
        min_clips=min_clips
    )
    
    if result:
        click.echo()
        click.echo("=" * 60)
        click.echo(f"✅ Compilation created successfully!")
        click.echo(f"📁 Output: {result}")
        click.echo("=" * 60)
    else:
        click.echo()
        click.echo("❌ Failed to create compilation")


if __name__ == '__main__':
    create_weekly_compilation()

