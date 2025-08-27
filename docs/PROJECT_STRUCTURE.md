# 📁 Clipppy Project Structure

## 🏗️ Repository Organization

```
clipppy-2/
├── 📂 src/                          # Core Application Code
│   ├── __init__.py
│   └── clip_enhancer.py             # Video enhancement with captions
│
├── 📂 services/                     # Background Services
│   ├── __init__.py
│   ├── tiktok_uploader.py          # TikTok upload queue & rate limiting
│   └── dashboard.py                # Web dashboard for monitoring
│
├── 📂 config/                       # Configuration Files
│   ├── config.yaml                 # Main system configuration
│   └── streamers.yaml              # Legacy streamer config (deprecated)
│
├── 📂 docs/                         # Documentation
│   ├── README.md                   # Main project documentation
│   ├── PROJECT_STRUCTURE.md        # This file
│   └── clipppy.drawio.xml          # System architecture diagram
│
├── 📂 tests/                        # Test Files
│   └── (test files go here)
│
├── 📂 clips/                        # Generated Video Content
│   ├── temp/                       # Temporary processing files
│   ├── fonts/                      # Font files for captions
│   └── *.mp4                       # Generated clips
│
├── 📂 uploads/                      # TikTok Upload Queue
│   ├── jynxzi_clippy/              # Per-account upload folders
│   │   ├── README.txt              # Upload instructions
│   │   ├── YYYYMMDD_HHMMSS_video.mp4
│   │   └── YYYYMMDD_HHMMSS_caption.txt
│   └── {streamer}_clippy/          # Additional streamer accounts
│
├── 📂 data/                         # Application Data
│   └── upload_history.json         # Upload tracking and analytics
│
├── 📂 logs/                         # System Logs
│   └── clipppy.log                 # Application logs
│
├── 🎬 twitch_clip_bot.py           # Main CLI Application
├── 📋 requirements.txt             # Python dependencies
└── 📄 .gitignore                   # Git ignore rules
```

## 📦 Module Descriptions

### 🎯 **Core Application** (`twitch_clip_bot.py`)
- Main CLI interface with all commands
- Stream monitoring and spike detection
- Twitch API integration for clip creation
- Orchestrates all services and components

### 🎬 **Source Code** (`src/`)
- **`clip_enhancer.py`**: Video processing pipeline
  - WhisperX transcription with word-level timing
  - MrBeast-style caption generation
  - TikTok format conversion (1080x1920)
  - MoviePy-based video composition

### ⚙️ **Services** (`services/`)
- **`tiktok_uploader.py`**: Upload management
  - Rate limiting (posts per day)
  - Posting time windows
  - Upload queue organization
  - Caption generation with hashtags

- **`dashboard.py`**: Web monitoring interface
  - Real-time performance tracking
  - Upload statistics across accounts
  - System health monitoring
  - Streamer management controls

### 🔧 **Configuration** (`config/`)
- **`config.yaml`**: Complete system configuration
  - Twitch API credentials
  - TikTok account settings
  - Streamer configurations and thresholds
  - Enhancement styles and preferences

## 🚀 **Usage Patterns**

### **Development**
```bash
# Run from project root
python twitch_clip_bot.py config        # Check configuration
python twitch_clip_bot.py list-streamers # View streamer setup
python twitch_clip_bot.py dashboard     # Start web interface
```

### **Production**
```bash
# Monitor all enabled streamers
python twitch_clip_bot.py start

# Monitor specific streamer
python twitch_clip_bot.py start --streamer jynxzi
```

### **File Paths**
- All imports use relative paths from project root
- Configuration files are in `config/` folder
- Generated content is organized in dedicated folders
- No hardcoded paths outside project directory

## 🔄 **Data Flow**

1. **`twitch_clip_bot.py`** monitors streams and detects spikes
2. **`src/clip_enhancer.py`** processes and enhances clips
3. **`services/tiktok_uploader.py`** queues uploads with rate limiting
4. **`services/dashboard.py`** provides monitoring and analytics
5. **`config/config.yaml`** drives all system behavior

## 🛠️ **Maintenance**

- **Add new streamers**: Edit `config/config.yaml`
- **New enhancement styles**: Update style library in config
- **Monitor uploads**: Check `uploads/` folders
- **View logs**: Check `logs/clipppy.log`
- **Analytics**: Access web dashboard at `localhost:8080`

This structure provides clear separation of concerns, easy maintenance, and professional organization for scaling the Phase 1 system.
