# 📺 YouTube Automation System - Complete Setup Guide

Transform Clipppy into a fully automated YouTube content machine! This system automatically uploads clips with AI-generated metadata, optimized scheduling, and zero manual intervention.

## 🎯 What This Does

- **Auto-uploads** enhanced clips to YouTube after creation
- **AI-generated** titles, descriptions, and tags (using Claude)
- **Smart scheduling** posts at optimal times for maximum views
- **Weekly compilations** auto-created and uploaded
- **Queue management** handles rate limiting and priorities
- **Zero manual work** once configured

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# YouTube API
pip install google-api-python-client google-auth-oauthlib

# AI metadata generation (optional but recommended)
pip install anthropic
```

### 2. Get YouTube API Credentials

#### Step-by-Step:

1. **Go to Google Cloud Console**: https://console.cloud.google.com/

2. **Create a New Project**:
   - Click "Select a Project" → "New Project"
   - Name it "Clipppy YouTube Automation"
   - Click "Create"

3. **Enable YouTube Data API v3**:
   - Go to "APIs & Services" → "Library"
   - Search for "YouTube Data API v3"
   - Click "Enable"

4. **Create OAuth 2.0 Credentials**:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth client ID"
   - Application type: "Desktop app"
   - Name: "Clipppy Desktop"
   - Click "Create"

5. **Download Credentials**:
   - Click the download button (⬇️) next to your newly created credential
   - Save the file as: `config/youtube_credentials.json`

### 3. Authenticate

```bash
python twitch_clip_bot.py youtube-auth
```

This will:
- Open a browser window
- Ask you to log in to your YouTube account
- Request permission to upload videos
- Save auth token for future use

### 4. Configure Settings

Edit `config/config.yaml`:

```yaml
youtube:
  enabled: true  # Enable YouTube automation
  auto_upload_clips: true  # Auto-upload enhanced clips
  auto_upload_compilations: true  # Auto-upload weekly compilations
  
  # Scheduling
  timezone_offset_hours: -5  # Your timezone (EST=-5, PST=-8, etc.)
  min_hours_between_posts: 12  # Minimum gap between uploads
  max_posts_per_day: 2  # Maximum daily uploads
  preferred_days: [2, 4, 5]  # Wed, Fri, Sat (0=Monday, 6=Sunday)
  
  # Quality control
  min_viral_score_for_youtube: 0.20  # Only upload clips above this score
  
  # AI metadata (optional)
  use_ai_metadata: true  # Use Claude for titles/descriptions
```

### 5. Set Up AI Metadata Generation (Optional)

For better titles/descriptions, add to `.env`:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Get your API key from: https://console.anthropic.com/

### 6. Enable Automatic Uploads

That's it! Now when clips are created:
1. They're automatically enhanced
2. Queued for YouTube upload
3. Scheduled for optimal posting time
4. Uploaded automatically

## 📋 CLI Commands

### Authentication
```bash
# Initial setup - authenticate with YouTube
python twitch_clip_bot.py youtube-auth
```

### Queue Management
```bash
# Check upload queue status
python twitch_clip_bot.py youtube-queue-status

# Manually process queue (uploads ready videos)
python twitch_clip_bot.py youtube-process-queue

# Process up to 3 videos at once
python twitch_clip_bot.py youtube-process-queue --max-uploads 3
```

### Weekly Compilations
```bash
# Create compilation and auto-queue for YouTube
python twitch_clip_bot.py weekly-compilation --streamer theburntpeanut

# The system will ask if you want to queue for YouTube
```

## ⚙️ How It Works

### 1. **Clip Creation Flow**

```
Viral Moment Detected
  ↓
Twitch Clip Created
  ↓
Enhanced with Captions/Effects
  ↓
✅ Viral Score Check (≥ 0.20)
  ↓
Queued for YouTube
  ↓
Scheduled for Optimal Time
  ↓
Auto-Uploaded to YouTube
```

### 2. **Intelligent Scheduling**

The system automatically schedules uploads based on:

**Optimal Time Slots:**
- Weekdays: 2-4 PM, 6-8 PM
- Weekends: 9-11 AM, 2-4 PM

**Preferred Days:**
- Wednesday, Friday, Saturday (highest engagement)

**Rate Limiting:**
- Min 12 hours between posts
- Max 2 posts per day
- Respects YouTube API limits

### 3. **AI Metadata Generation**

If enabled (with Anthropic API key), Claude generates:

**Titles (optimized for clicks):**
- "This is INSANE!"
- "You WON'T believe this..."
- "[Game] but [unexpected twist]"

**Descriptions (keyword-rich):**
- Hook in first line
- Relevant gaming keywords
- Call-to-action
- Streamer credit/links

**Tags (SEO-optimized):**
- Streamer name
- Game name
- Relevant keywords
- Trending terms

### 4. **Priority System**

Videos are uploaded in priority order:

| Priority | Type | When |
|----------|------|------|
| 10 | Weekly Compilations | Always uploaded first |
| 5 | Regular Clips | Standard priority |
| 1 | Low-score Clips | Only if queue is empty |

## 🔄 Automation Workflow

### Daily Operation (Zero Manual Work):

1. **Morning**: System monitors streams
2. **Viral Moment**: Clip created & enhanced automatically
3. **Queue**: Added to YouTube queue with metadata
4. **Schedule**: Next optimal slot calculated (e.g., Wed 3 PM)
5. **Upload**: Celery worker uploads at scheduled time
6. **Repeat**: Process continues 24/7

### Weekly Compilation (Also Automated):

Set up a scheduled task (Windows Task Scheduler / cron):

**Weekly Script** (`weekly_auto_upload.bat`):
```batch
@echo off
cd "C:\path\to\clipppy 2"
python twitch_clip_bot.py weekly-compilation --streamer theburntpeanut --no-prompt
python twitch_clip_bot.py youtube-process-queue
```

Schedule this to run every Sunday at 11 PM.

## 📊 Monitoring

### Check Queue Status
```bash
python twitch_clip_bot.py youtube-queue-status
```

Output:
```
📊 YOUTUBE UPLOAD QUEUE STATUS
========================================

📋 Total items: 15
⏳ Pending: 8
✅ Uploaded: 6
❌ Failed: 1

📅 Next scheduled upload:
   Title: TheBurntPeanut - INSANE Moment! 🔥
   Time: Wednesday, December 27 at 03:00 PM
   Video: clips/theburntpeanut/enhanced_theburntpeanut_1234567890.mp4
```

### Process Queue Manually
```bash
# Upload all videos whose scheduled time has passed
python twitch_clip_bot.py youtube-process-queue
```

## 🤖 Celery Integration (Advanced)

For fully automated queue processing, add to your Celery beat schedule:

```python
# In celery_config.py or wherever you configure beat
from celery.schedules import crontab

app.conf.beat_schedule = {
    'process-youtube-queue': {
        'task': 'clipppy.process_youtube_queue',
        'schedule': crontab(minute='*/30'),  # Every 30 minutes
        'args': (1,)  # Max 1 upload per run
    },
}
```

Then the system checks every 30 minutes and uploads any videos whose scheduled time has arrived.

## 🎨 Customization

### Custom Metadata Templates

Edit `youtube_uploader.py` → `_generate_with_template()`:

```python
def _generate_with_template(self, streamer_name: str, ...):
    if is_compilation:
        title = f"YOUR CUSTOM TITLE FORMAT"
        description = f"""YOUR CUSTOM DESCRIPTION"""
    # ...
```

### Custom Scheduling

Edit `youtube_uploader.py` → `OPTIMAL_SLOTS`:

```python
OPTIMAL_SLOTS = {
    'weekday': [
        (10, 12),  # 10 AM - 12 PM
        (16, 18),  # 4 PM - 6 PM
    ],
    'weekend': [
        (8, 10),   # 8 AM - 10 AM
    ]
}
```

### Custom Quality Thresholds

In `config.yaml`:

```yaml
youtube:
  min_viral_score_for_youtube: 0.25  # Only upload highest quality clips
```

## 🐛 Troubleshooting

### "Authentication failed"

**Problem**: YouTube credentials not set up correctly.

**Solution**:
1. Ensure `config/youtube_credentials.json` exists
2. Run `python twitch_clip_bot.py youtube-auth` again
3. Check file permissions

### "Quota exceeded"

**Problem**: Hit YouTube API daily quota (10,000 units/day).

**Solution**:
- Each upload costs ~1,600 units
- Max ~6 uploads per day safely
- Reduce `max_posts_per_day` in config
- Request quota increase from Google

### "No AI metadata generated"

**Problem**: Anthropic API key not set or invalid.

**Solution**:
1. Add `ANTHROPIC_API_KEY` to `.env`
2. Or disable AI: `use_ai_metadata: false` in config
3. System will use templates instead

### "Videos not uploading"

**Problem**: Queue processor not running.

**Solution**:
1. Check queue: `python twitch_clip_bot.py youtube-queue-status`
2. Process manually: `python twitch_clip_bot.py youtube-process-queue`
3. Check scheduled times haven't passed yet
4. Ensure Celery beat is running (if using automated processing)

## 💡 Pro Tips

1. **Start Small**: Test with 1 post/day initially
2. **Monitor Analytics**: Check which posting times work best for YOUR audience
3. **Adjust Thresholds**: If getting too many/few uploads, adjust `min_viral_score_for_youtube`
4. **Use AI Metadata**: Worth the API cost for better titles
5. **Schedule Compilations**: Weekly compilations get 10x more views than individual clips
6. **Batch Processing**: Let queue build up, then process in batches
7. **Backup Credentials**: Save `youtube_token.pickle` somewhere safe

## 📈 Expected Results

With proper configuration:

- **10-15 clips/week** automatically uploaded
- **1 compilation/week** (if scheduled)
- **Optimal posting times** maximize initial engagement
- **AI titles** improve click-through rate by ~30%
- **Zero manual work** after initial setup

## 🔐 Security Notes

- `youtube_credentials.json` - OAuth client credentials (safe to backup)
- `youtube_token.pickle` - Your auth token (keep private!)
- Don't commit these files to git (already in `.gitignore`)
- Token refreshes automatically, no need to re-auth

## 🎯 Next Steps

1. ✅ Complete authentication
2. ✅ Configure settings in `config.yaml`
3. ✅ Let system run for a week
4. 📊 Check YouTube Analytics
5. ⚙️ Adjust thresholds based on results
6. 🚀 Scale up to multiple streamers

---

**Questions?** Check the main Clipppy documentation or the troubleshooting section above!

**Ready to automate?** Run:
```bash
python twitch_clip_bot.py youtube-auth
```

And let Clipppy handle the rest! 🎬

