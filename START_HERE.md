# 🎯 CLIPPPY - 24/7 QUICK START

## ✅ STATUS: READY TO RUN

**Everything is configured. Just start it.**

---

## 🚀 TO START:

```cmd
START_ALL.bat
```

**Leave all windows open.**

---

## 📊 WHAT IT DOES:

1. **Waits for theburntpeanut to go live** (checks every 30 seconds)
2. **Monitors chat for viral moments** (smart algorithm)
3. **Creates clips automatically** (~5-8 quality clips per day)
4. **Enhances with captions/effects** (Celery workers)
5. **Saves raw clips** (for weekly compilations)
6. **Uploads to YouTube** (smart scheduling)

---

## 🛡️ IT'S BULLETPROOF:

- ✅ **Crash protection** - Errors are logged, system keeps running
- ✅ **Auto-restart listeners** - When peanut goes live, spawns new listener
- ✅ **Controller stays running** - Even when streamers go offline
- ✅ **Only theburntpeanut enabled** - No other streamers

---

## 📁 IMPORTANT FILES:

- `START_ALL.bat` - Run this to start everything
- `STOP_ALL.bat` - Run this to stop everything
- `config/config.yaml` - All settings (thresholds, streamers, YouTube)
- `logs/always_on_controller.log` - Main system log
- `logs/listener_theburntpeanut.log` - Clip detection log

---

## 🎛️ CURRENT CONFIGURATION:

**Streamer:** Only theburntpeanut

**Thresholds (More Selective):**
- Viral score: 0.30
- Min chatters: 35
- Cooldown: 15 minutes
- Min viewers: 15

**YouTube:**
- Auto-upload: Enabled
- Posting: Every 4 hours, max 6/day
- Titles: Unique templates (AI disabled)

---

## 📖 DOCUMENTATION:

- `README_24_7.md` - This file (quick start)
- `YOUTUBE_AUTOMATION_GUIDE.md` - YouTube setup details
- `WEEKLY_COMPILATION_GUIDE.md` - How to create compilations
- `REDIS_CELERY_QUICKSTART.md` - Technical details

---

## 🔧 TO ADJUST SETTINGS:

Edit `config/config.yaml`:

```yaml
# Make clips more/less selective
score_threshold: 0.30  # Higher = fewer, better quality clips

# Change cooldown between clips
cooldown_seconds: 900  # 15 minutes

# Adjust YouTube posting
min_hours_between_posts: 4  # Time between uploads
max_posts_per_day: 6  # Daily limit
```

---

## 📺 WEEKLY COMPILATIONS (AUTOMATED):

### One-Time Setup:
```cmd
setup_weekly_task.bat
```

This creates a Windows scheduled task that runs every **Sunday at 11:59 PM** to:
1. Gather **ALL clips** from the past week (raw unenhanced clips)
2. Create a landscape (16:9) compilation video
3. Upload to YouTube automatically

**After setup: Fully automated. Nothing to do.**

### How It Works:
- Raw clips are saved automatically when created ✅
- Every Sunday at 11:59 PM, task runs automatically ✅
- Creates compilation from ALL clips (no limit) ✅
- Uses raw Twitch clips (landscape 16:9, no conversion needed) ✅
- Uploads as regular YouTube video (not a short) ✅

### Manual Run (Optional):
```cmd
python twitch_clip_bot.py weekly-compilation --streamer theburntpeanut --days 7 --yes
```

---

## ❓ TROUBLESHOOTING:

**No clips being created:**
- Check `logs/listener_theburntpeanut.log`
- Look for "rejected" messages
- Lower `score_threshold` to 0.25 if too strict

**Controller crashed:**
- Check `logs/always_on_controller.log` for errors
- Restart with `START_ALL.bat`
- Should auto-recover now with crash protection

**YouTube not uploading:**
```cmd
python twitch_clip_bot.py youtube-queue-status
python twitch_clip_bot.py youtube-process-queue
```

---

## ✅ FINAL CHECKLIST:

- [x] Only theburntpeanut enabled
- [x] Thresholds raised for quality
- [x] Crash protection added
- [x] YouTube automation working
- [x] Weekly compilations working
- [x] Raw clips being saved

**Ready to run 24/7!**

---

## 🎯 SUMMARY:

**Just run `START_ALL.bat` and leave it.**

The system will:
- Monitor theburntpeanut's stream 24/7
- Create quality clips automatically
- Enhance and upload to YouTube
- Keep running even if errors occur

**That's it. It's ready.** 🚀

