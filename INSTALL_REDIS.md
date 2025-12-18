# 📦 Installing Redis on Windows

Redis is required for Clipppy's distributed task queue. Here's how to install it:

---

## Method 1: Chocolatey (Easiest)

### Step 1: Install Chocolatey (if not installed)
Open PowerShell as Administrator and run:
```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

### Step 2: Install Redis
```bash
choco install redis-64
```

### Step 3: Verify
```bash
redis-server --version
```

---

## Method 2: Manual Download

### Step 1: Download Redis
Go to: https://github.com/microsoftarchive/redis/releases

Download: `Redis-x64-3.0.504.msi` (or latest version)

### Step 2: Install
1. Run the MSI installer
2. Accept default options
3. Check "Add to PATH" if prompted

### Step 3: Verify
Open Command Prompt:
```bash
redis-server --version
```

---

## Method 3: Memurai (Commercial, Free Tier Available)

Memurai is a Windows-native Redis alternative:

### Step 1: Download
Go to: https://www.memurai.com/get-memurai

### Step 2: Install
Run the installer and follow prompts

### Step 3: Configure
Memurai works as a drop-in Redis replacement. No code changes needed!

---

## Testing Redis

### Start Redis Server
```bash
redis-server
```

You should see:
```
                _._
           _.-``__ ''-._
      _.-``    `.  `_.  ''-._           Redis 3.0.504 (00000000/0) 64 bit
  .-`` .-```.  ```\/    _.,_ ''-._
 (    '      ,       .-`  | `,    )     Running in standalone mode
 |`-._`-...-` __...-.``-._|'` _.-'|     Port: 6379
 |    `-._   `._    /     _.-'    |     PID: 1234
  `-._    `-._  `-./  _.-'    _.-'
 |`-._`-._    `-.__.-'    _.-'_.-'|
 |    `-._`-._        _.-'_.-'    |           http://redis.io
  `-._    `-._`-.__.-'_.-'    _.-'
 |`-._`-._    `-.__.-'    _.-'_.-'|
 |    `-._`-._        _.-'_.-'    |
  `-._    `-._`-.__.-'_.-'    _.-'
      `-._    `-.__.-'    _.-'
          `-._        _.-'
              `-.__.-'

[1234] 18 Dec 12:00:00.000 # Server started, Redis version 3.0.504
[1234] 18 Dec 12:00:00.000 * The server is now ready to accept connections on port 6379
```

### Test Connection
Open a new terminal:
```bash
redis-cli ping
```

Should return: `PONG`

---

## Troubleshooting

### "redis-server: command not found"

**Fix:**
1. Make sure Redis is installed
2. Add Redis to PATH:
   - Open System Properties → Environment Variables
   - Edit PATH
   - Add: `C:\Program Files\Redis` (or your install location)
3. Restart terminal

### Port 6379 Already in Use

**Fix:**
```bash
# Kill existing Redis
taskkill /F /IM redis-server.exe

# Or use a different port
redis-server --port 6380
```

Then update `celery_tasks.py`:
```python
app = Celery(
    'clipppy',
    broker='redis://localhost:6380/0',  # Changed port
    backend='redis://localhost:6380/1'
)
```

### Redis Crashes on Startup

**Fix:**
1. Check if another Redis instance is running
2. Delete Redis data files:
   - `C:\Program Files\Redis\dump.rdb`
   - `C:\Program Files\Redis\appendonly.aof`
3. Restart Redis

---

## Redis as a Windows Service (Optional)

To run Redis automatically on startup:

### Using redis-server (from MSI installer)
The MSI installer should set this up automatically.

### Manual Service Setup
```bash
# Install as service
redis-server --service-install

# Start service
redis-server --service-start

# Stop service
redis-server --service-stop

# Uninstall service
redis-server --service-uninstall
```

---

## Next Steps

Once Redis is installed and running:

1. Run `START_ALL.bat` to start Clipppy
2. Open http://localhost:5555 for Flower dashboard
3. Watch clips get enhanced automatically!

---

## Need Help?

- Redis Docs: https://redis.io/docs/
- Windows Redis: https://github.com/microsoftarchive/redis
- Memurai: https://www.memurai.com/

---

## Quick Reference

```bash
# Start Redis
redis-server

# Test connection
redis-cli ping

# Check version
redis-server --version

# Monitor Redis activity
redis-cli monitor

# View all keys
redis-cli keys *

# Clear all data (careful!)
redis-cli FLUSHALL
```

---

That's it! Redis is now ready for Clipppy. 🚀

