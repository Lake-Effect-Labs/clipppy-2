# YouTube Interval-Based Scheduling

## 🎯 Overview

The YouTube upload system now uses **interval-based scheduling** instead of optimal time slots. Videos are posted at regular intervals (default: 2 hours) starting immediately, making it much simpler and more predictable.

## ✅ What Changed

### Before (Optimal Slots)
- Videos scheduled for specific "optimal" times (9 AM, 2 PM, etc.)
- Only posted on preferred days (Wed, Fri, Sat)
- Complex logic to find next available slot
- Could delay uploads by days

### After (Interval-Based)
- Videos posted every X hours (configurable, default: 2 hours)
- Starts immediately (5 minutes from queuing)
- Simple, predictable scheduling
- Queue processes continuously until empty

## 📋 Configuration

Edit `config/config.yaml`:

```yaml
youtube:
  enabled: true
  min_hours_between_posts: 2  # Post every 2 hours
  max_posts_per_day: 12       # Up to 12 posts per day
  auto_upload_clips: true
  # ... other settings
```

## 🚀 Usage

### Reschedule Existing Queue

If you have videos scheduled with the old system, reschedule them:

```bash
python reschedule_youtube_queue.py
```

This will:
- Update all pending videos to use interval-based scheduling
- Start posting in 5 minutes
- Space videos by the configured interval (2 hours default)

### Manual Processing

Process the queue manually at any time:

```bash
python process_youtube_queue.py
```

This will:
- Show queue status
- Authenticate with YouTube
- Upload any videos scheduled for now or earlier
- Process up to 10 videos per run

### Automatic Processing (Recommended)

The Celery worker automatically processes the queue every hour:

```bash
# Make sure Celery worker is running
python -m celery -A celery_tasks worker --loglevel=info --pool=solo
```

The periodic task `process-youtube-queue-hourly` runs every hour and uploads any videos that are due.

## 📊 Example Schedule

If you queue 3 videos at 12:00 PM with 2-hour intervals:

- Video 1: 12:05 PM (5 minutes from now)
- Video 2: 2:05 PM (2 hours later)
- Video 3: 4:05 PM (2 hours later)

## 🔧 How It Works

### When Videos Are Queued

1. System checks the last scheduled video time
2. Adds the configured interval (e.g., 2 hours)
3. Schedules the new video for that time
4. If no previous videos, starts in 5 minutes

### When Queue Is Processed

1. Loads all pending videos
2. Checks if scheduled time has passed
3. Uploads videos that are due
4. Marks them as uploaded
5. Stops when no more due videos or max uploads reached

## 📝 Scripts Reference

### `reschedule_youtube_queue.py`
- Reschedules all pending videos with interval-based times
- Run once when switching from optimal slots to intervals
- Safe to run multiple times (just reschedules again)

### `process_youtube_queue.py`
- Manually processes the upload queue
- Shows detailed status and progress
- Useful for immediate uploads or testing

### `youtube_uploader.py`
- Core uploader class
- Can be run directly to test authentication
- Contains all scheduling and upload logic

## 🎛️ Customization

### Change Upload Interval

Edit `config/config.yaml`:

```yaml
youtube:
  min_hours_between_posts: 3  # Post every 3 hours instead
```

Then reschedule existing videos:

```bash
python reschedule_youtube_queue.py
```

### Change Automatic Processing Frequency

Edit `celery_tasks.py`:

```python
'process-youtube-queue-hourly': {
    'task': 'clipppy.process_youtube_queue',
    'schedule': 1800.0,  # Every 30 minutes instead of 1 hour
    'kwargs': {'max_uploads': 10}
},
```

## 🐛 Troubleshooting

### Videos Not Uploading

1. Check if Celery worker is running
2. Verify YouTube authentication: `python youtube_uploader.py`
3. Check scheduled times: `python process_youtube_queue.py`
4. Look for errors in Celery logs

### Authentication Issues

1. Make sure `config/youtube_credentials.json` exists
2. Delete `config/youtube_token.pickle` to re-authenticate
3. Run `python youtube_uploader.py` to authenticate manually

### Queue Not Processing

1. Check if scheduled times are in the future
2. Reschedule if needed: `python reschedule_youtube_queue.py`
3. Process manually: `python process_youtube_queue.py`

## 📈 Benefits

✅ **Simpler**: No complex optimal time slot logic  
✅ **Faster**: Starts uploading immediately  
✅ **Predictable**: Know exactly when videos will post  
✅ **Flexible**: Easy to adjust intervals  
✅ **Reliable**: Continuous processing until queue is empty  

## 🔄 Migration from Old System

If you were using the optimal slots system:

1. Update your code (already done)
2. Update config: `min_hours_between_posts: 2`
3. Reschedule pending videos: `python reschedule_youtube_queue.py`
4. Restart Celery worker to pick up new schedule
5. Monitor first few uploads to verify

That's it! Your YouTube uploads will now use simple interval-based scheduling.

