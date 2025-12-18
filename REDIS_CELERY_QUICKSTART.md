# 🚀 Redis + Celery Quick Start

## Install Redis (One-Time Setup)

### Windows
```bash
# Option 1: Chocolatey
choco install redis-64

# Option 2: Download MSI
# https://github.com/microsoftarchive/redis/releases
```

### Verify
```bash
redis-server --version
```

---

## Start Clipppy (Every Time)

### Easy Mode: One-Click Start
```bash
START_ALL.bat
```
Opens 4 windows: Redis, Celery Worker, Flower Dashboard, Controller

### Manual Mode: Individual Services

**Terminal 1: Redis**
```bash
start_redis.bat
```

**Terminal 2: Celery Worker**
```bash
start_celery_worker.bat
```

**Terminal 3: Flower Dashboard (Optional)**
```bash
start_flower.bat
```

**Terminal 4: Controller**
```bash
python launch_always_on.py
```

---

## Monitor Tasks

### Flower Dashboard
Open: http://localhost:5555

See:
- Active tasks
- Completed tasks
- Failed tasks
- Worker status
- Queue length

### Logs
- Celery: `logs/celery_worker.log`
- Controller: `logs/always_on_controller.log`
- Listeners: `logs/listener_[streamer].log`

---

## How It Works

1. **Listener** detects viral moment → Creates clip
2. **Listener** sends clip to **Redis queue**
3. **Celery worker** picks up task from queue
4. **Worker** downloads & enhances clip
5. **Enhanced clip** saved to `clips/[streamer]/enhanced/`
6. **You** manually post to TikTok

---

## Troubleshooting

### Redis Not Found
```bash
# Install Redis first (see above)
# Add to PATH
# Restart terminal
```

### Can't Connect to Redis
```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# If not, start Redis
start_redis.bat
```

### Tasks Not Processing
1. Check Redis is running: `redis-cli ping`
2. Check Celery worker terminal for errors
3. Check logs: `logs/celery_worker.log`

---

## Resume Bullets (Updated)

```
• Engineered an AI-powered Twitch automation system that monitors 40K+ viewer 
  streams in real-time, detects viral moments using multi-signal analysis, and 
  autonomously generates TikTok-ready clips with AI captions—reducing manual 
  editing time by 95%

• Architected distributed video processing pipeline using Redis and Celery for 
  fault-tolerant task queuing with automatic retry logic and real-time monitoring 
  dashboard, processing 10+ viral clips daily across multiple concurrent streams
```

---

## What's Different?

### Before (File-Based Queue)
- 3 local worker threads
- File-based communication
- No monitoring
- No retry logic
- TikTok auto-posting

### After (Redis + Celery)
- 1 Celery worker (scalable to N)
- Redis message queue
- Flower monitoring dashboard
- Automatic retry logic
- Manual TikTok posting (simpler)

---

## Scale Up (Optional)

Want faster processing? Add more workers:

```bash
# Terminal 5: Worker 2
celery -A celery_tasks worker --loglevel=info --concurrency=1 --pool=solo --hostname=worker2@%h

# Terminal 6: Worker 3
celery -A celery_tasks worker --loglevel=info --concurrency=1 --pool=solo --hostname=worker3@%h
```

Each worker = 1 concurrent enhancement. 3 workers = 3x throughput!

---

## That's It! 🎉

Run `START_ALL.bat` and you're good to go!

