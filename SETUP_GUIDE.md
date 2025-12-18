# 🚀 Clipppy Setup Guide - Redis + Celery Edition

## Overview

Clipppy now uses **Redis + Celery** for distributed video processing! This provides:
- ✅ **Fault-tolerant** task queuing
- ✅ **Automatic retry** logic
- ✅ **Real-time monitoring** via Flower dashboard
- ✅ **Horizontal scaling** (add more workers as needed)
- ✅ **Production-ready** architecture

---

## 📋 Prerequisites

### 1. Install Redis

**Option A: Chocolatey (Recommended)**
```bash
choco install redis-64
```

**Option B: Manual Download**
1. Download from: https://github.com/microsoftarchive/redis/releases
2. Extract to `C:\Program Files\Redis`
3. Add to PATH: `C:\Program Files\Redis`

**Verify Installation:**
```bash
redis-server --version
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `celery` - Distributed task queue
- `redis` - Redis Python client
- `flower` - Monitoring dashboard
- All existing dependencies

---

## 🎯 Quick Start

### Option 1: Start Everything at Once (Easiest)

Double-click `START_ALL.bat`

This opens 4 windows:
1. **Redis Server** - Message broker
2. **Celery Worker** - Processes enhancement tasks
3. **Flower Dashboard** - Monitoring UI at http://localhost:5555
4. **Always-On Controller** - Stream monitoring

### Option 2: Start Services Manually

**Terminal 1: Start Redis**
```bash
start_redis.bat
```

**Terminal 2: Start Celery Worker**
```bash
start_celery_worker.bat
```

**Terminal 3: Start Flower Dashboard (Optional)**
```bash
start_flower.bat
```

**Terminal 4: Start Controller**
```bash
python launch_always_on.py
```

---

## 🏗️ New Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   ALWAYS-ON CONTROLLER                       │
│  - Monitors which streamers are live                         │
│  - Spawns/kills listener processes                           │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  LISTENER 1  │      │  LISTENER 2  │      │  LISTENER N  │
│              │      │              │      │              │
│ - Monitors   │      │ - Monitors   │      │ - Monitors   │
│   chat       │      │   chat       │      │   chat       │
│ - Detects    │      │ - Detects    │      │ - Detects    │
│   viral      │      │   viral      │      │   viral      │
│ - Creates    │      │ - Creates    │      │ - Creates    │
│   clip       │      │   clip       │      │   clip       │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       │  Sends to Redis     │                     │
       └─────────────────────┼─────────────────────┘
                             ▼
              ┌──────────────────────────────┐
              │         REDIS QUEUE          │
              │  (persistent, fault-tolerant)│
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │      CELERY WORKER           │
              │  (1 worker, auto-retry)      │
              │                              │
              │  - Downloads clip            │
              │  - Enhances with AI          │
              │  - Saves to folder           │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────┐
              │   clips/streamer/enhanced/   │
              │   (ready for manual upload)  │
              └──────────────────────────────┘
                             
              ┌──────────────────────────────┐
              │   FLOWER DASHBOARD           │
              │   http://localhost:5555      │
              │   (monitor tasks in real-time)│
              └──────────────────────────────┘
```

---

## 🎛️ Monitoring

### Flower Dashboard

Open http://localhost:5555 to see:
- ✅ Active tasks
- ✅ Completed tasks
- ✅ Failed tasks (with retry info)
- ✅ Worker status
- ✅ Task execution times
- ✅ Queue length

### Logs

- **Celery Worker**: `logs/celery_worker.log`
- **Controller**: `logs/always_on_controller.log`
- **Listeners**: `logs/listener_[streamer].log`

---

## 🔧 Configuration

### Celery Settings (celery_tasks.py)

```python
task_time_limit=1800           # 30 min max per task
task_soft_time_limit=1500      # 25 min soft limit
worker_prefetch_multiplier=1   # Only fetch 1 task at a time
task_acks_late=True            # Fault tolerance
```

### Redis Connection

Default: `redis://localhost:6379/0`

To change, edit `celery_tasks.py`:
```python
app = Celery(
    'clipppy',
    broker='redis://your-redis-host:6379/0',
    backend='redis://your-redis-host:6379/1'
)
```

---

## 🐛 Troubleshooting

### Redis Not Starting

**Error:** `redis-server: command not found`

**Fix:**
1. Install Redis (see Prerequisites)
2. Add to PATH
3. Restart terminal

### Celery Worker Errors

**Error:** `Cannot connect to Redis`

**Fix:**
1. Make sure Redis is running: `redis-cli ping` (should return `PONG`)
2. Check Redis logs in the Redis terminal

### Tasks Not Processing

**Check:**
1. Is Redis running? → `redis-cli ping`
2. Is Celery worker running? → Check terminal
3. Is there an error in logs? → Check `logs/celery_worker.log`

### Flower Dashboard Not Loading

**Fix:**
```bash
# Kill any existing Flower processes
taskkill /F /IM python.exe /FI "COMMANDLINE eq *flower*"

# Restart Flower
start_flower.bat
```

---

## 📊 Scaling

### Add More Workers

Want faster processing? Start additional workers:

```bash
# Terminal 5: Second worker
celery -A celery_tasks worker --loglevel=info --concurrency=1 --pool=solo --hostname=worker2@%h

# Terminal 6: Third worker
celery -A celery_tasks worker --loglevel=info --concurrency=1 --pool=solo --hostname=worker3@%h
```

Each worker can process 1 clip at a time. With 3 workers, you can process 3 clips simultaneously!

---

## 🎯 What Changed?

### Removed:
- ❌ 3 local enhancement worker threads
- ❌ TikTok auto-posting queue
- ❌ File-based queue system (`data/enhancement_queue/`)

### Added:
- ✅ Redis message broker
- ✅ Celery distributed task queue
- ✅ Flower monitoring dashboard
- ✅ Automatic retry logic
- ✅ Fault-tolerant architecture

### Simplified:
- ✅ 1 Celery worker instead of 3 threads
- ✅ Manual TikTok posting (clips saved to folders)
- ✅ Cleaner architecture

---

## 📝 Manual TikTok Posting

Enhanced clips are saved to:
```
clips/[streamer_name]/enhanced/
```

To post manually:
1. Open TikTok app
2. Navigate to enhanced clips folder
3. Select clip to upload
4. Add caption and hashtags
5. Post!

---

## 🚀 Resume-Worthy Features

This architecture demonstrates:
- ✅ **Distributed Systems** - Redis + Celery message queue
- ✅ **Fault Tolerance** - Automatic retry, persistent queue
- ✅ **Horizontal Scaling** - Add more workers as needed
- ✅ **Production Architecture** - Industry-standard tech stack
- ✅ **Real-time Monitoring** - Flower dashboard
- ✅ **Event-Driven Design** - Async task processing

---

## 📚 Additional Resources

- **Celery Docs**: https://docs.celeryproject.org/
- **Redis Docs**: https://redis.io/docs/
- **Flower Docs**: https://flower.readthedocs.io/

---

## 🎉 You're Ready!

Run `START_ALL.bat` and watch the magic happen! 🚀

