"""
Process YouTube Upload Queue

Manually process the YouTube upload queue to upload any videos
that are scheduled for now or earlier.
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')

from youtube_uploader import YouTubeUploader
from datetime import datetime

def main():
    print("=" * 60)
    print("YouTube Upload Queue Processor")
    print("=" * 60)
    print()
    
    # Initialize uploader
    uploader = YouTubeUploader()
    
    # Show current status
    status = uploader.get_queue_status()
    print(f"Queue Status:")
    print(f"   Total videos: {status['total']}")
    print(f"   Pending: {status['pending']}")
    print(f"   Uploaded: {status['uploaded']}")
    print(f"   Failed: {status['failed']}")
    print()
    
    if status['pending'] == 0:
        print("No pending videos in queue")
        return
    
    # Show next video
    if status['next_upload']:
        next_video = status['next_upload']
        scheduled = datetime.fromisoformat(next_video['scheduled_time'])
        now = datetime.now()
        time_until = (scheduled - now).total_seconds() / 60
        
        title = next_video['metadata']['title']
        # Remove emojis for Windows console
        title_clean = ''.join(c for c in title if ord(c) < 0x10000)
        
        print(f"Next video: {title_clean}")
        print(f"Scheduled: {scheduled.strftime('%I:%M %p')}")
        
        if time_until > 0:
            print(f"Time until: {int(time_until)} minutes")
        else:
            print(f"Ready to upload (overdue by {int(abs(time_until))} minutes)")
        print()
    
    # Authenticate
    print("Authenticating with YouTube...")
    if not uploader.authenticate():
        print("ERROR: Authentication failed!")
        print("Run 'python youtube_uploader.py' to set up authentication")
        return
    print("Authenticated successfully!")
    print()
    
    # Process queue
    print("Processing queue...")
    print("-" * 60)
    uploader.process_queue(max_uploads=10)  # Process up to 10 videos
    print("-" * 60)
    print()
    
    # Show updated status
    status = uploader.get_queue_status()
    print(f"Updated Queue Status:")
    print(f"   Pending: {status['pending']}")
    print(f"   Uploaded: {status['uploaded']}")
    print(f"   Failed: {status['failed']}")
    print()
    print("Done!")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

