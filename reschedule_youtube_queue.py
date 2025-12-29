"""
Reschedule YouTube Upload Queue

This script reschedules all pending YouTube uploads to use interval-based scheduling
instead of optimal time slots. Videos will be posted every X hours starting now.
"""

import sys
import os
from datetime import datetime

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')

from youtube_uploader import YouTubeUploader

def main():
    print("Rescheduling YouTube upload queue...")
    print()
    
    # Initialize uploader
    uploader = YouTubeUploader()
    
    # Show current status
    status = uploader.get_queue_status()
    print(f"Current Queue Status:")
    print(f"   Total videos: {status['total']}")
    print(f"   Pending: {status['pending']}")
    print(f"   Uploaded: {status['uploaded']}")
    print(f"   Failed: {status['failed']}")
    print()
    
    if status['pending'] == 0:
        print("No pending videos to reschedule")
        return
    
    # Reschedule pending videos
    uploader.reschedule_pending_videos()
    print()
    
    # Show updated status
    print("New Schedule:")
    queue = uploader._load_queue()
    pending = [q for q in queue if q['status'] == 'pending']
    
    for i, item in enumerate(pending, 1):
        scheduled = datetime.fromisoformat(item['scheduled_time'])
        title = item['metadata']['title'][:60]
        # Remove emojis from title for Windows console
        title_clean = ''.join(c for c in title if ord(c) < 0x10000)
        print(f"   {i}. {scheduled.strftime('%I:%M %p')} - {title_clean}...")
    
    print()
    print("Rescheduling complete!")
    print()
    print("To process the queue, run:")
    print("   python youtube_uploader.py")
    print()
    print("Or set up automatic processing with:")
    print("   python -c \"from celery_tasks import process_youtube_queue_task; process_youtube_queue_task.apply_async()\"")

if __name__ == '__main__':
    main()

