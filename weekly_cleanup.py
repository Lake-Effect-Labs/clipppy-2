"""
Weekly Cleanup Script - Saves Disk Space

Runs after weekly compilation to delete:
- Enhanced clips (already uploaded to YouTube)
- Old compilation videos (already uploaded)
- Keeps RAW clips for future compilations if needed
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def cleanup_old_files(streamer_name='theburntpeanut', keep_days=14):
    """
    Clean up old files to save disk space
    
    Args:
        streamer_name: Streamer to clean up files for
        keep_days: Keep files from last N days (default: 14 days)
    """
    logger.info("🧹 Starting weekly cleanup...")
    
    clips_dir = Path(f"clips/{streamer_name}")
    
    if not clips_dir.exists():
        logger.warning(f"⚠️ Clips directory not found: {clips_dir}")
        return
    
    cutoff_date = datetime.now() - timedelta(days=keep_days)
    deleted_count = 0
    freed_space = 0
    
    # 1. Delete old enhanced clips (already uploaded)
    logger.info(f"🗑️ Deleting enhanced clips older than {keep_days} days...")
    for enhanced_clip in clips_dir.glob("enhanced_*.mp4"):
        try:
            # Get file modification time
            mod_time = datetime.fromtimestamp(enhanced_clip.stat().st_mtime)
            
            if mod_time < cutoff_date:
                size = enhanced_clip.stat().st_size
                enhanced_clip.unlink()
                deleted_count += 1
                freed_space += size
                logger.info(f"   Deleted: {enhanced_clip.name}")
        except Exception as e:
            logger.warning(f"   Failed to delete {enhanced_clip.name}: {e}")
    
    # 2. Delete old compilation videos (already uploaded)
    compilations_dir = clips_dir / "compilations"
    if compilations_dir.exists():
        logger.info(f"🗑️ Deleting compilation videos older than {keep_days} days...")
        for compilation in compilations_dir.glob("*.mp4"):
            try:
                mod_time = datetime.fromtimestamp(compilation.stat().st_mtime)
                
                if mod_time < cutoff_date:
                    size = compilation.stat().st_size
                    compilation.unlink()
                    deleted_count += 1
                    freed_space += size
                    logger.info(f"   Deleted: {compilation.name}")
            except Exception as e:
                logger.warning(f"   Failed to delete {compilation.name}: {e}")
    
    # 3. Optionally clean up old RAW clips (if you want to save even more space)
    # Uncomment this if you want to delete raw clips after 30 days
    # raw_clips_dir = clips_dir / "raw"
    # if raw_clips_dir.exists():
    #     logger.info(f"🗑️ Deleting raw clips older than 30 days...")
    #     raw_cutoff = datetime.now() - timedelta(days=30)
    #     for raw_clip in raw_clips_dir.glob("raw_*.mp4"):
    #         try:
    #             mod_time = datetime.fromtimestamp(raw_clip.stat().st_mtime)
    #             if mod_time < raw_cutoff:
    #                 size = raw_clip.stat().st_size
    #                 raw_clip.unlink()
    #                 deleted_count += 1
    #                 freed_space += size
    #                 logger.info(f"   Deleted: {raw_clip.name}")
    #         except Exception as e:
    #             logger.warning(f"   Failed to delete {raw_clip.name}: {e}")
    
    # Summary
    freed_mb = freed_space / (1024 * 1024)
    freed_gb = freed_mb / 1024
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✅ Cleanup complete!")
    logger.info(f"   Files deleted: {deleted_count}")
    if freed_gb >= 1:
        logger.info(f"   Space freed: {freed_gb:.2f} GB")
    else:
        logger.info(f"   Space freed: {freed_mb:.2f} MB")
    logger.info("=" * 60)

if __name__ == "__main__":
    cleanup_old_files()

