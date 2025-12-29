# YouTube Queue Fix - December 26, 2025

## Problem
Clips were being enhanced but NOT queued for YouTube upload, even though `youtube.auto_upload_clips` was enabled and `min_viral_score_for_youtube` was set to 0.15.

## Root Causes

### Issue 1: Wrong Config Path
**File**: `celery_tasks.py` line 173

**Bug**: The code was reading YouTube config from `streamer_config.get('youtube', {})` instead of the global config.

```python
# OLD (BROKEN)
youtube_config = streamer_config.get('youtube', {})  # This doesn't exist!
min_score = youtube_config.get('min_viral_score_for_youtube', 0.20)
```

**Fix**: Load the global config directly:

```python
# NEW (FIXED)
from twitch_clip_bot import TwitchClipBot
bot = TwitchClipBot('config/config.yaml')
youtube_config = bot.config.get('youtube', {})
min_score = youtube_config.get('min_viral_score_for_youtube', 0.15)
```

### Issue 2: Multiple Celery Workers
**Problem**: Multiple Celery worker processes were running - some with old buggy code, some with new fixed code.

**PIDs Found**:
- PID 21316 (started 20:58) - OLD CODE
- PID 26984 (started 23:26) - NEW CODE

Tasks were randomly picked up by either worker. If the old worker got it, YouTube queueing failed silently.

**Fix**: Killed all old workers, ensured only one worker with updated code is running.

### Issue 3: Listener Not Reloaded
**Problem**: The listener process (`twitch_clip_bot.py start`) was still running with OLD code that imported the OLD version of `celery_tasks.py`.

**Fix**: Killed listener process (PID 24104) and restarted it to load the new code.

## Verification Steps

1. ✅ Fixed `celery_tasks.py` to read from global YouTube config
2. ✅ Killed old Celery worker (PID 21316)
3. ✅ Killed old listener (PID 24104)
4. ✅ Restarted listener with new code
5. ⏳ **NEXT**: Wait for new clip to verify it gets queued for YouTube

## Expected Behavior (After Fix)

When a clip is created with `viral_score >= 0.15`:

1. Listener sends clip to Celery queue
2. Celery worker enhances the clip
3. After enhancement, worker checks: `if viral_score >= min_viral_score_for_youtube`
4. If true, calls `upload_to_youtube_task.apply_async()` to queue for YouTube
5. Clip appears in `data/youtube_upload_queue.json` with status "pending"

**Celery worker log should show**:
```
📺 Queuing for YouTube (viral score: 0.XX)
```

**OR if score too low**:
```
⏭️ Skipping YouTube (viral score 0.XX < 0.15)
```

## Configuration

**File**: `config/config.yaml`

```yaml
youtube:
  enabled: true
  auto_upload_clips: true
  min_viral_score_for_youtube: 0.15  # Lowered from 0.20
```

## Files Modified

1. `celery_tasks.py` - Line 173-178: Fixed config loading
2. `config/config.yaml` - Line 23: Lowered threshold from 0.20 to 0.15

## Processes Restarted

1. Celery worker (killed PID 21316, kept PID 26984)
2. Listener for theburntpeanut (killed PID 24104, respawned)

---

**Status**: Fixed and waiting for next clip to verify
**Date**: December 26, 2025 23:43

