# ✅ YouTube Automation - Complete Setup

## 🎯 What's Automated

When you run `START_ALL.bat`, the system now automatically:

1. ✅ **Monitors Twitch streams** for viral moments
2. ✅ **Creates clips** when viral moments are detected
3. ✅ **Enhances clips** with captions, effects, and cropping
4. ✅ **Queues clips for YouTube** (if viral score ≥ 0.15)
5. ✅ **Uploads to YouTube every hour** automatically
6. ✅ **Posts every 2 hours** until queue is empty

## 🚀 Quick Start

### First Time Setup

1. **Authenticate with YouTube** (one-time):
   ```bash
   python youtube_uploader.py
   ```
   This will open a browser for OAuth authentication.

2. **Start all services**:
   ```bash
   START_ALL.bat
   ```

That's it! Everything runs automatically now.

## 📊 What Services Are Running

When you run `START_ALL.bat`, these windows open:

| Service | Purpose | Window Title |
|---------|---------|--------------|
| **Redis** | Message queue for tasks | Clipppy - Redis |
| **Celery Worker** | Processes enhancement tasks | Clipppy - Celery Worker |
| **Celery Beat** | Schedules YouTube uploads every hour | Clipppy - Celery Beat |
| **Flower** | Web dashboard at http://localhost:5555 | Clipppy - Flower |
| **Controller** | Monitors streamers and creates clips | Clipppy - Controller |

## 🔄 How YouTube Automation Works

### Automatic Flow

```
Viral Moment Detected
    ↓
Clip Created
    ↓
Clip Enhanced (Celery Worker)
    ↓
Added to YouTube Queue (if viral_score ≥ 0.15)
    ↓
Celery Beat checks queue every hour
    ↓
Uploads videos that are scheduled for now
    ↓
Next video scheduled 2 hours later
```

### Scheduling Logic

- **First video**: Uploads 5 minutes after being queued
- **Subsequent videos**: Upload every 2 hours
- **Automatic processing**: Every hour, Celery Beat checks for due uploads
- **No manual intervention needed**: Runs 24/7

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
youtube:
  enabled: true                      # Enable YouTube uploads
  min_hours_between_posts: 2         # Post every 2 hours
  max_posts_per_day: 12              # Maximum daily uploads
  auto_upload_clips: true            # Auto-queue enhanced clips
  min_viral_score_for_youtube: 0.15  # Minimum score to queue
  use_ai_metadata: true              # Use AI for titles/descriptions
```

## 📋 Manual Controls

### Check Queue Status
```bash
python process_youtube_queue.py
```

Shows:
- How many videos are pending
- When the next upload is scheduled
- Upload history

### Reschedule Queue
```bash
python reschedule_youtube_queue.py
```

Useful if you:
- Change the interval in config
- Want to reset upload times
- Need to start uploading immediately

### Force Upload Now
```bash
python process_youtube_queue.py
```

This will:
- Authenticate with YouTube
- Upload any videos scheduled for now or earlier
- Process up to 10 videos at once

## 🔍 Monitoring

### Flower Dashboard
Open http://localhost:5555 to see:
- Active tasks (enhancement, uploads)
- Task history and success rates
- Worker status
- Queue sizes

### Celery Beat Window
Watch the "Clipppy - Celery Beat" window to see:
- When periodic tasks run
- YouTube queue processing logs
- Scheduling information

### Controller Window
Watch the "Clipppy - Controller" window to see:
- Viral moments detected
- Clips created
- Enhancement status
- YouTube queueing

## 🎛️ Customization

### Change Upload Frequency

**Every 3 hours instead of 2:**
```yaml
# config/config.yaml
youtube:
  min_hours_between_posts: 3
```

Then reschedule: `python reschedule_youtube_queue.py`

### Change Processing Frequency

**Check queue every 30 minutes instead of hourly:**

Edit `celery_tasks.py`:
```python
'process-youtube-queue-hourly': {
    'task': 'clipppy.process_youtube_queue',
    'schedule': 1800.0,  # 30 minutes (in seconds)
    'kwargs': {'max_uploads': 10}
},
```

Restart services: `STOP_ALL.bat` then `START_ALL.bat`

### Change Viral Score Threshold

**Only upload clips with score ≥ 0.25:**
```yaml
# config/config.yaml
youtube:
  min_viral_score_for_youtube: 0.25
```

## 🐛 Troubleshooting

### Videos Not Uploading

**Check Celery Beat is running:**
- Look for "Clipppy - Celery Beat" window
- Should show periodic task execution logs

**Check authentication:**
```bash
python youtube_uploader.py
```

**Check queue:**
```bash
python process_youtube_queue.py
```

### Authentication Expired

Delete the token and re-authenticate:
```bash
del config\youtube_token.pickle
python youtube_uploader.py
```

### Queue Not Processing

**Restart Celery Beat:**
1. Close "Clipppy - Celery Beat" window
2. Run: `start_celery_beat.bat`

**Or restart everything:**
```bash
STOP_ALL.bat
START_ALL.bat
```

### Videos Scheduled Too Far in Future

Reschedule with current intervals:
```bash
python reschedule_youtube_queue.py
```

## 📁 Important Files

| File | Purpose |
|------|---------|
| `youtube_uploader.py` | Core upload logic |
| `celery_tasks.py` | Task definitions and scheduling |
| `start_celery_beat.bat` | Starts the scheduler |
| `data/youtube_upload_queue.json` | Upload queue (JSON) |
| `config/youtube_credentials.json` | OAuth credentials |
| `config/youtube_token.pickle` | OAuth token (auto-generated) |

## 🎉 Success Indicators

You know it's working when:

✅ All 5 service windows are open  
✅ Celery Beat shows "Scheduler: Sending due task..."  
✅ Flower dashboard shows completed upload tasks  
✅ Videos appear on your YouTube channel  
✅ Queue status shows "uploaded" videos  

## 📈 Expected Behavior

With default settings:
- **Clips created**: ~10 per day (across all streamers)
- **YouTube uploads**: ~6-8 per day (viral score ≥ 0.15)
- **Upload frequency**: Every 2 hours
- **Processing check**: Every hour
- **Fully automated**: No manual intervention needed

## 🔒 Security Notes

- OAuth tokens are stored locally in `config/youtube_token.pickle`
- Never commit `youtube_credentials.json` or `youtube_token.pickle` to git
- Tokens expire after ~7 days of inactivity (will auto-refresh)
- If token expires, just run `python youtube_uploader.py` to re-authenticate

## 💡 Pro Tips

1. **Monitor first 24 hours** - Watch Flower dashboard to ensure tasks complete
2. **Check YouTube Studio** - Verify videos are uploading correctly
3. **Adjust viral threshold** - If too many/few videos, adjust `min_viral_score_for_youtube`
4. **Use Flower for debugging** - Shows detailed task logs and errors
5. **Keep services running** - Use Task Scheduler to auto-start on boot

---

## 🎊 You're All Set!

Just run `START_ALL.bat` and the system will:
- Monitor streams 24/7
- Create viral clips automatically
- Upload to YouTube every 2 hours
- Handle everything without manual intervention

**Check your YouTube channel in a few hours to see the magic! 🚀**

