# YouTube Automation - Quick Start

## ⚡ 2-Step Setup

### Step 1: Authenticate (One Time Only)
```bash
python youtube_uploader.py
```
- Opens browser for Google OAuth
- Grants YouTube upload permission
- Saves token for future use

### Step 2: Start Everything
```bash
START_ALL.bat
```
- Opens 5 windows (Redis, Celery Worker, Celery Beat, Flower, Controller)
- Runs 24/7 automatically
- No further action needed!

---

## ✅ What Happens Automatically

```
Stream Monitoring → Viral Detection → Clip Creation → Enhancement → YouTube Queue → Auto Upload
    (24/7)           (AI-powered)      (Twitch API)    (Captions)     (Every 2hrs)   (Every hour)
```

---

## 📊 Quick Status Check

### View Queue
```bash
python process_youtube_queue.py
```

### View Dashboard
Open: http://localhost:5555

---

## 🎛️ Key Settings

**File:** `config/config.yaml`

```yaml
youtube:
  min_hours_between_posts: 2         # Upload every 2 hours
  min_viral_score_for_youtube: 0.15  # Only clips with score ≥ 0.15
  max_posts_per_day: 12              # Daily upload limit
```

---

## 🔄 Common Commands

| Task | Command |
|------|---------|
| **Start all services** | `START_ALL.bat` |
| **Stop all services** | `STOP_ALL.bat` |
| **Check queue status** | `python process_youtube_queue.py` |
| **Reschedule uploads** | `python reschedule_youtube_queue.py` |
| **View dashboard** | Open http://localhost:5555 |

---

## 🎉 That's It!

The system now runs completely automatically:
- ✅ Monitors Twitch streams 24/7
- ✅ Creates clips when viral moments happen
- ✅ Enhances with captions and effects
- ✅ Uploads to YouTube every 2 hours
- ✅ No manual work required!

**Just run `START_ALL.bat` and let it run!**

