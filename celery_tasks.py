#!/usr/bin/env python3
"""
Celery Tasks for Clipppy - Distributed Video Enhancement Pipeline
==================================================================

Handles asynchronous clip enhancement using Redis as message broker.
Provides automatic retry logic, task monitoring, and fault tolerance.
"""

import os
import sys
import logging
from pathlib import Path
from celery import Celery, Task
from celery.signals import task_prerun, task_postrun, task_failure
from typing import Dict, Optional
import json

# Add project root to Python path for imports
# This ensures Celery worker can find modules like clip_enhancer_v2
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables
def load_env_file():
    """Load environment variables from .env file"""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env_file()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/celery_worker.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Celery app
app = Celery(
    'clipppy',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

# Celery configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=1800,  # 30 minutes max per task
    task_soft_time_limit=1500,  # 25 minute soft limit
    worker_prefetch_multiplier=1,  # Only fetch 1 task at a time (since enhancement is heavy)
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks to prevent memory leaks
    task_acks_late=True,  # Only ack task after completion (fault tolerance)
    task_reject_on_worker_lost=True,  # Requeue if worker dies
    result_expires=3600,  # Keep results for 1 hour
)

# Task signals for monitoring
@task_prerun.connect
def task_prerun_handler(task_id, task, *args, **kwargs):
    """Log when task starts"""
    logger.info(f"🚀 Starting task: {task.name} (ID: {task_id})")

@task_postrun.connect
def task_postrun_handler(task_id, task, *args, **kwargs):
    """Log when task completes"""
    logger.info(f"✅ Completed task: {task.name} (ID: {task_id})")

@task_failure.connect
def task_failure_handler(task_id, exception, *args, **kwargs):
    """Log when task fails"""
    logger.error(f"❌ Task failed: {task_id} - {exception}")


class EnhancementTask(Task):
    """Custom task class with retry logic and error handling"""
    
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 3}
    retry_backoff = True  # Exponential backoff
    retry_backoff_max = 600  # Max 10 minutes between retries
    retry_jitter = True  # Add randomness to prevent thundering herd


@app.task(base=EnhancementTask, bind=True, name='clipppy.enhance_clip')
def enhance_clip_task(self, clip_data: Dict) -> Dict:
    """
    Celery task for enhancing a Twitch clip.
    
    Args:
        clip_data: Dictionary containing:
            - clip_url: URL of the clip to enhance
            - streamer_name: Display name of streamer
            - streamer_handle: Twitch username
            - game_name: Game being played
            - reason: Why this clip was created
            - streamer_config: Full streamer configuration
            
    Returns:
        Dictionary with enhancement results:
            - success: bool
            - output_path: Path to enhanced clip
            - error: Error message if failed
            - telemetry: Enhancement metrics
    """
    try:
        # Import here to avoid issues with multiprocessing
        # Ensure project root is in sys.path (should already be there from module init)
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Debug: Log current working directory and sys.path
        logger.info(f"📂 Current working directory: {os.getcwd()}")
        logger.info(f"📂 Project root: {project_root}")
        logger.info(f"📂 sys.path[0]: {sys.path[0] if sys.path else 'empty'}")
        
        # Change working directory to project root to ensure all imports work
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        try:
            from clip_enhancer_v2 import ClipEnhancerV2
            logger.info("✅ Successfully imported ClipEnhancerV2")
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
        
        clip_url = clip_data.get('clip_url')
        streamer_name = clip_data.get('streamer_name')
        streamer_config = clip_data.get('streamer_config', {})
        
        logger.info(f"🎬 Processing clip for {streamer_name}: {clip_url}")
        
        # Initialize enhancer
        enhancer = ClipEnhancerV2('config/config.yaml')
        
        # Download clip
        logger.info(f"📥 Downloading clip...")
        clip_path = enhancer.download_clip(clip_url)
        
        if not clip_path:
            raise Exception("Failed to download clip")
        
        # Enhance clip
        logger.info(f"🎨 Enhancing clip...")
        output_path, telemetry = enhancer.enhance_clip(
            clip_path=clip_path,
            streamer_config=streamer_config
        )
        
        if not output_path:
            raise Exception("Enhancement failed")
        
        logger.info(f"✅ Successfully enhanced clip: {output_path}")
        
        # Auto-queue for YouTube if enabled
        youtube_config = streamer_config.get('youtube', {})
        if youtube_config.get('enabled', False) and youtube_config.get('auto_upload_clips', True):
            try:
                # Check if clip meets minimum viral score threshold
                viral_score = clip_data.get('viral_score', 0)
                min_score = youtube_config.get('min_viral_score_for_youtube', 0.20)
                
                if viral_score >= min_score:
                    logger.info(f"📺 Queuing for YouTube (viral score: {viral_score:.2f})")
                    
                    # Queue YouTube upload as separate task
                    upload_to_youtube_task.apply_async(args=[{
                        'video_path': output_path,
                        'streamer_name': streamer_name,
                        'clip_context': {
                            'game_name': clip_data.get('game_name', 'Gaming'),
                            'reason': clip_data.get('reason', ''),
                            'viral_score': viral_score,
                            'clip_url': clip_url
                        },
                        'is_compilation': False,
                        'priority': youtube_config.get('clip_priority', 5)
                    }])
                else:
                    logger.info(f"⏭️ Skipping YouTube (viral score {viral_score:.2f} < {min_score})")
            except Exception as e:
                logger.warning(f"⚠️ Failed to queue YouTube upload: {e}")
        
        # Return results
        return {
            'success': True,
            'output_path': output_path,
            'clip_url': clip_url,
            'streamer_name': streamer_name,
            'telemetry': {
                'processing_time_ms': telemetry.processing_time_ms,
                'words_rendered': telemetry.words_rendered,
                'emphasis_hits': telemetry.emphasis_hits,
                'lufs_before': telemetry.lufs_before,
                'lufs_after': telemetry.lufs_after,
                'cuts_count': telemetry.cuts_count,
                'ms_removed': telemetry.ms_removed,
                'hook_used': telemetry.hook_used
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Enhancement failed: {e}")
        
        # Log retry attempt
        if self.request.retries < self.max_retries:
            logger.warning(f"⏳ Retrying in {self.retry_backoff_max}s (attempt {self.request.retries + 1}/{self.max_retries})")
        
        # Return error result
        return {
            'success': False,
            'error': str(e),
            'clip_url': clip_data.get('clip_url'),
            'streamer_name': clip_data.get('streamer_name')
        }


@app.task(name='clipppy.health_check')
def health_check_task() -> Dict:
    """Simple health check task for monitoring"""
    return {
        'status': 'healthy',
        'worker': 'clipppy-enhancement-worker'
    }


@app.task(name='clipppy.cleanup_temp_files')
def cleanup_temp_files_task() -> Dict:
    """Periodic task to clean up temporary files"""
    try:
        import shutil
        from datetime import datetime, timedelta
        
        temp_dir = Path('clips/temp')
        if not temp_dir.exists():
            return {'cleaned': 0}
        
        cleaned = 0
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Remove files older than 24 hours
        for file_path in temp_dir.rglob('*'):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    file_path.unlink()
                    cleaned += 1
        
        logger.info(f"🧹 Cleaned up {cleaned} temporary files")
        return {'cleaned': cleaned}
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return {'error': str(e)}


@app.task(name='clipppy.upload_to_youtube')
def upload_to_youtube_task(video_data: Dict) -> Dict:
    """
    Celery task for uploading a video to YouTube.
    
    Args:
        video_data: Dictionary containing:
            - video_path: Path to the video file
            - streamer_name: Name of streamer
            - clip_context: Optional context about the clip
            - is_compilation: Whether this is a compilation
            - priority: Upload priority (1-10)
            
    Returns:
        Dictionary with upload results
    """
    try:
        from youtube_uploader import YouTubeUploader
        
        video_path = video_data.get('video_path')
        streamer_name = video_data.get('streamer_name')
        clip_context = video_data.get('clip_context')
        is_compilation = video_data.get('is_compilation', False)
        priority = video_data.get('priority', 5)
        
        logger.info(f"📺 Queuing YouTube upload for {streamer_name}: {video_path}")
        
        # Initialize uploader
        uploader = YouTubeUploader()
        
        # Add to queue (will handle scheduling automatically)
        uploader.queue_video(
            video_path=video_path,
            streamer_name=streamer_name,
            clip_context=clip_context,
            is_compilation=is_compilation,
            priority=priority
        )
        
        return {
            'success': True,
            'video_path': video_path,
            'status': 'queued'
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ YouTube queuing failed: {error_msg}")
        return {
            'success': False,
            'error': error_msg
        }


@app.task(name='clipppy.process_youtube_queue')
def process_youtube_queue_task(max_uploads: int = 1) -> Dict:
    """
    Celery task to process the YouTube upload queue.
    Run this periodically (e.g., every hour) to check for scheduled uploads.
    
    Args:
        max_uploads: Maximum number of videos to upload in this run
        
    Returns:
        Dictionary with processing results
    """
    try:
        from youtube_uploader import YouTubeUploader
        
        logger.info("📺 Processing YouTube upload queue...")
        
        uploader = YouTubeUploader()
        uploader.process_queue(max_uploads=max_uploads)
        
        # Get queue status
        status = uploader.get_queue_status()
        
        return {
            'success': True,
            'queue_status': status
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ YouTube queue processing failed: {error_msg}")
        return {
            'success': False,
            'error': error_msg
        }


# Periodic task schedule (optional)
app.conf.beat_schedule = {
    'cleanup-temp-files-daily': {
        'task': 'clipppy.cleanup_temp_files',
        'schedule': 86400.0,  # Every 24 hours
    },
}


if __name__ == '__main__':
    # Start worker from command line
    app.start()

