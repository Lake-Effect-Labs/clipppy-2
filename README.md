# 🎬 Clipppy - Phase 1: TikTok Automation Pipeline

**Automatically creates and uploads viral clips from Twitch streams to TikTok accounts**

Clipppy is an intelligent automation system that monitors Twitch streams for hype moments, creates clips when metrics spike, enhances them with MrBeast-style captions and effects, and uploads them to TikTok accounts with proper rate limiting and scheduling.

## 🚀 Quick Start

### 1. **Setup**
```bash
# Install dependencies
python setup.py

# Or manually:
pip install -r requirements.txt
```

### 2. **Configure**
```bash
# Edit configuration file
# Add your Twitch API credentials and streamer settings
notepad config/config.yaml
```

### 3. **Run**
```bash
# Check configuration
python clipppy.py config

# Start monitoring all enabled streamers
python clipppy.py start

# Open web dashboard
python clipppy.py dashboard
```

## 📁 Project Structure

```
clipppy-2/
├── 🎬 clipppy.py                    # Main entry point
├── 🎯 twitch_clip_bot.py           # Core CLI application
├── 🎨 clip_enhancer.py             # Video enhancement & captions
├── 🧠 viral_detector.py            # Advanced viral detection
├── 📱 tiktok_uploader.py           # Upload queue & rate limiting
├── 📊 dashboard.py                 # Web monitoring interface
├── 📂 config/                      # Configuration files
│   └── config.yaml                 # Main system configuration
├── 📂 docs/                        # Documentation
│   ├── README.md                   # Detailed documentation
│   └── PROJECT_STRUCTURE.md        # Complete structure guide
├── 📂 tests/                       # Test files
├── 📂 clips/                       # Generated video content
├── 📂 uploads/                     # TikTok upload queue
├── 📂 data/                        # Application data & analytics
└── 📂 logs/                        # System logs
```

See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) for complete details.

## 🎯 Phase 1 Features

### ✅ **Complete Automation Pipeline**
- **Stream Monitoring**: Real-time monitoring of multiple Twitch streamers
- **Spike Detection**: Automated clip creation when chat/viewer spikes occur
- **AI Enhancement**: Professional captions with word-level timing using WhisperX
- **TikTok Upload**: Automated queuing and uploading to TikTok accounts
- **Performance Dashboard**: Web-based monitoring and analytics

### 🎨 **Professional Video Enhancement**
- **MrBeast-Style Captions**: Large, animated text with viral emphasis
- **Word-Level Timing**: Perfect synchronization using WhisperX
- **TikTok Format**: Automatic conversion to 1080x1920 vertical format
- **Viral Elements**: Emojis, emphasis colors, and dynamic animations
- **Multiple Styles**: Configurable enhancement profiles (MrBeast, minimal, gaming, viral)

### 📱 **TikTok Integration**
- Automated upload queuing with caption generation
- Smart rate limiting (configurable posts per day)
- Posting time windows to maximize engagement
- Hashtag management per streamer
- Performance tracking and analytics

## 🎮 CLI Commands

### **Main Operations**
```bash
python clipppy.py start                     # Monitor all enabled streamers
python clipppy.py start --streamer jynxzi   # Monitor specific streamer
python clipppy.py dashboard                 # Start web dashboard
```

### **Management**
```bash
python clipppy.py config                    # Check system status
python clipppy.py list-streamers           # View all streamers
python clipppy.py toggle-streamer jynxzi   # Enable/disable streamer
python clipppy.py stats --days 7           # Performance statistics
```

### **Manual Operations**
```bash
python clipppy.py testclip                 # Create test clip
python clipppy.py upload video.mp4 --streamer jynxzi
python clipppy.py enhance https://clips.twitch.tv/xyz --vertical
```

## 📊 Example Output

### **Successful Caption Generation**
```
🎬 TESTING PHASE 1 CAPTION SYSTEM
✅ Input video: clips/test_clip.mp4 (5.3 MB)
🎤 WhisperX transcribed 48 words with perfect alignment
📝 Created 16 synchronized caption phrases
🔥 Added viral emphasis to 2 key moments
📱 Converted to TikTok format: 1080x1920
✅ Video queued for TikTok upload!
🎉 PHASE 1 COMPLETE!
```

### **System Status**
```
🔧 Configuration Status:
   Twitch Client ID: ✅ Set
   Twitch Client Secret: ✅ Set
   Twitch OAuth Token: ⚠️ Missing (needed for clips)

📺 Streamers:
   Enabled: 1
     🟢 jynxzi → @jynxzi_clippy
   Disabled: 1
     🔴 shroud

📱 TikTok Integration: ✅ Ready
🎬 Clip Enhancement: ✅ Ready
```

## 🔧 Configuration

### **Main Config** (`config/config.yaml`)
```yaml
# Twitch API Credentials
twitch:
  client_id: "your_twitch_client_id"
  client_secret: "your_twitch_client_secret"
  oauth_token: "your_twitch_oauth_token"

# Streamers Configuration
streamers:
  - name: "jynxzi"
    twitch_username: "jynxzi"
    broadcaster_id: "85018793"
    enabled: true
    
    # TikTok Account Settings
    tiktok_account:
      username: "jynxzi_clippy"
      max_posts_per_day: 3
      hashtags: ["#jynxzi", "#rainbowsix", "#gaming", "#viral", "#fyp"]
    
    # Enhancement Settings
    enhancement:
      style: "mrbeast"
      vertical_format: true
      add_captions: true
```

### **Get Twitch Credentials**
1. Go to [Twitch Developer Console](https://dev.twitch.tv/console/apps)
2. Create a new application
3. Note your Client ID and Client Secret
4. Get OAuth token: `python clipppy.py oauth-help --generate-url`

## 🎬 End-to-End Workflow

1. **Stream Detection**: Monitor configured streamers for live status
2. **Spike Detection**: Analyze chat messages/second and viewer growth
3. **Clip Creation**: Use Twitch API to create 30-second clips
4. **Enhancement Pipeline**:
   - Download clip with yt-dlp
   - Transcribe audio with WhisperX (word-level timing)
   - Generate viral captions with emphasis and emojis
   - Convert to TikTok 1080x1920 format
   - Render final video with MoviePy
5. **Upload Queue**: Add to TikTok upload queue with proper rate limiting
6. **Analytics**: Track performance and update dashboard

## 🌐 Web Dashboard

Access the monitoring dashboard at `http://localhost:8080`:

- **Real-time Performance**: Upload stats across all accounts
- **System Health**: Monitoring active services and error rates
- **Streamer Controls**: Enable/disable streamers
- **Analytics**: View estimated revenue and engagement metrics

## 🎯 Scaling & Business Model

**Current Capacity:**
- 10 streamers × 5 clips/day = 1,500 clips/month
- Target: 10.2M views/month per account
- Estimated revenue: $5K-$10K/month at scale

**Revenue Sources:**
- TikTok Creator Fund payments (~$0.40 per 1K views)
- Potential sponsorship integration
- Future: Direct streamer service offerings

## 🔧 Development

### **Testing**
```bash
# Test complete system
python tests/test_organized_system.py

# Test specific components
python -m pytest tests/
```

### **Contributing**
1. Follow the organized folder structure
2. Use relative imports from project root
3. Update documentation for new features
4. Test with `python tests/test_organized_system.py`

## 📄 Documentation

- [`docs/README.md`](docs/README.md) - Complete feature documentation
- [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) - Detailed structure guide
- [`docs/clipppy.drawio.xml`](docs/clipppy.drawio.xml) - System architecture diagram

## 🚧 Current Limitations (Phase 1)

- **TikTok API**: Manual upload queue (automatic API integration in Phase 2)
- **Single Instance**: One Clipppy instance per machine (containerization in Phase 2)
- **Basic Analytics**: Simulated revenue data (real TikTok analytics in Phase 2)

## 📈 Next Steps (Phase 2)

- Direct TikTok API integration for automatic uploads
- Multi-instance deployment with Docker
- Real-time TikTok analytics integration
- Advanced AI features (GPT-4 for captions, DALL-E for thumbnails)
- Streamer partnership program

---

**🎬 Ready to create viral content? Get started now!**

```bash
python setup.py    # Setup
python clipppy.py config    # Configure
python clipppy.py start     # Go viral! 🚀
```
