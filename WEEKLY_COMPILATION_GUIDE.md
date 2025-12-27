# 🎬 Weekly Compilation Feature

Create long-form YouTube highlight videos from your streamer's raw clips!

## 🚀 Quick Start

### Basic Usage

```bash
# Create a weekly compilation for theburntpeanut
python twitch_clip_bot.py weekly-compilation

# Specify different streamer
python twitch_clip_bot.py weekly-compilation --streamer jynxzi

# Look back 14 days instead of 7
python twitch_clip_bot.py weekly-compilation --days 14
```

### Advanced Options

```bash
# Limit to most recent 20 clips
python twitch_clip_bot.py weekly-compilation --max-clips 20

# Require at least 10 clips
python twitch_clip_bot.py weekly-compilation --min-clips 10

# Skip intro and transitions for faster rendering
python twitch_clip_bot.py weekly-compilation --no-intro --no-transitions

# Custom output location
python twitch_clip_bot.py weekly-compilation --output "my_compilation.mp4"
```

## 📁 How It Works

### 1. **Raw Clips Are Automatically Saved**

When Clipppy creates a clip, it now saves TWO versions:
- **Enhanced clip**: `clips/theburntpeanut/enhanced_theburntpeanut_1234567890.mp4` (for TikTok)
- **Raw clip**: `clips/theburntpeanut/raw/raw_theburntpeanut_1234567890.mp4` (for compilations)

### 2. **Compilation Structure**

The compilation video includes:
- **Intro** (3 seconds): "THEBURNTPEANUT WEEKLY HIGHLIGHTS"
- **Clips**: All raw clips in chronological order
- **Transitions** (1 second): "HIGHLIGHT #X" between clips
- **Automatic concatenation**: No gaps, smooth playback

### 3. **Output Location**

Compilations are saved to:
```
clips/
└── theburntpeanut/
    ├── raw/                    # Raw clips (source material)
    ├── enhanced_*.mp4          # TikTok clips
    └── compilations/           # Weekly compilations
        └── theburntpeanut_weekly_20251225.mp4
```

## 🎯 Use Cases

### Weekly YouTube Upload

```bash
# Every Sunday, create a compilation of last 7 days
python twitch_clip_bot.py weekly-compilation --streamer theburntpeanut

# Upload to YouTube with title: "TheBurntPeanut - Best Moments This Week"
```

### Monthly Mega-Compilation

```bash
# End of month, create a 30-day compilation
python twitch_clip_bot.py weekly-compilation --days 30 --streamer theburntpeanut
```

### Best-Of Selection

```bash
# Manually select top 15 clips from the past 2 weeks
python twitch_clip_bot.py weekly-compilation --days 14 --max-clips 15
```

## 📊 Preview Before Creating

The command shows a preview before rendering:

```
📊 Found 23 raw clips:
   1. raw_theburntpeanut_1735084803.mp4 - 28.5s - 2024-12-24 10:20
   2. raw_theburntpeanut_1735088403.mp4 - 31.2s - 2024-12-24 11:20
   ...
   23. raw_theburntpeanut_1735171203.mp4 - 29.8s - 2024-12-25 10:20

⏱️ Estimated compilation length: 11.2 minutes
🎬 Options: Intro=Yes, Transitions=Yes

📹 Create compilation from these clips? [Y/n]:
```

## ⚙️ Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--streamer` | `theburntpeanut` | Streamer name |
| `--days` | `7` | Days to look back |
| `--max-clips` | None | Maximum clips to include |
| `--min-clips` | None | Minimum clips required |
| `--no-intro` | False | Skip intro |
| `--no-transitions` | False | Skip transitions |
| `--output` | Auto | Custom output path |

## 🎨 Customization Ideas

Want to customize intro/transitions? Edit `weekly_compilation.py`:

```python
def create_intro_clip(self, streamer_name: str, duration: float = 3.0):
    # Customize your intro here
    # Add background music, graphics, etc.
```

## 🐛 Troubleshooting

### "No raw clips found"

**Problem**: The raw clips directory doesn't exist or is empty.

**Solution**: Raw clips are only saved for NEW clips created after this feature was added. Create some new clips first:

```bash
python twitch_clip_bot.py start --streamer theburntpeanut
```

Wait for a few viral moments to be detected and clipped. The raw versions will automatically be saved to `clips/theburntpeanut/raw/`.

### "MoviePy is required"

**Problem**: MoviePy library is not installed.

**Solution**:
```bash
pip install moviepy
```

### Rendering takes a long time

**Problem**: Video rendering is CPU-intensive.

**Solution**: 
- Use `--no-transitions` to skip transition generation
- Use `--max-clips` to limit the number of clips
- The script uses `preset='medium'` for good balance of speed/quality

### Out of memory errors

**Problem**: Too many clips loaded at once.

**Solution**:
```bash
# Limit clips to prevent memory issues
python twitch_clip_bot.py weekly-compilation --max-clips 20
```

## 💡 Pro Tips

1. **Run weekly on a schedule**: Use Windows Task Scheduler or cron to auto-create compilations every Sunday

2. **Test with fewer clips first**:
   ```bash
   python twitch_clip_bot.py weekly-compilation --max-clips 5 --no-transitions
   ```

3. **Archive old compilations**: After uploading to YouTube, move the compilation to an archive folder to keep things organized

4. **Combine with manual editing**: Use the generated compilation as a starting point, then edit in your favorite video editor for additional polish

5. **Add background music**: The compilations have the original audio. Add non-copyrighted music in post-production for better YouTube performance

## 🎯 Example Workflow

```bash
# Monday through Sunday: Clipppy automatically creates clips
python always_on_controller.py

# Sunday night: Create compilation
python twitch_clip_bot.py weekly-compilation --streamer theburntpeanut

# Output: clips/theburntpeanut/compilations/theburntpeanut_weekly_20251225.mp4

# Upload to YouTube with:
# Title: "TheBurntPeanut - Best Moments This Week!"
# Description: "This week's funniest and most epic moments..."
# Tags: gaming, theburntpeanut, highlights, funny moments
```

## 🔮 Future Enhancements

Potential features to add:
- Custom intro/outro videos
- Background music tracks
- Auto-generated titles/descriptions for YouTube
- Thumbnail generation
- Direct YouTube upload via API
- Automatic chapter markers
- Blur/censor profanity in compilation mode

---

**Questions or issues?** Check the main Clipppy documentation or modify `weekly_compilation.py` to fit your needs!

