# 🎨 Assets Directory

This directory contains all media assets used by the Auto-Enhancement Engine v2.

## 📁 Directory Structure

```
assets/
├── logos/           # Brand logos and watermarks  
├── fonts/           # Custom fonts for captions
├── mascot/          # Clippy mascot animations
├── sfx/             # Sound effects library
└── README.md        # This file
```

## 📝 Asset Requirements

### 🏷️ Logos (`logos/`)
- **Format**: PNG with transparency
- **Size**: 200x200px recommended
- **Files**:
  - `logo.png` - Main Clipppy logo
  - `watermark.png` - Simplified watermark version

### 🔤 Fonts (`fonts/`) 
- **Format**: TTF or OTF
- **Recommended fonts**:
  - `Montserrat-ExtraBold.ttf` - For viral captions
  - `Arial-Bold.ttf` - Fallback font
  - `BebasNeue-Regular.ttf` - For headers

### 🤖 Mascot (`mascot/`)
- **Format**: WebM/APNG with alpha channel or PNG sequence
- **Duration**: ≤ 1.2 seconds per animation
- **Size**: 200x200px max
- **Variants**:
  - `pop.webm` - Pop in/out animation  
  - `bounce.webm` - Bounce animation
  - `spin.webm` - Spin animation
  - `hype.webm` - Celebration animation

### 🔊 SFX (`sfx/`)
- **Format**: WAV or MP3
- **Duration**: 0.5-2.0 seconds
- **Volume**: Normalized to -14 LUFS
- **Types**:
  - `impact.wav` - Impact sound for emphasis
  - `whoosh.wav` - Transition sound
  - `pop.wav` - Pop sound for mascot
  - `hype.wav` - Celebration sound

## 🛠️ Usage in Presets

Assets are referenced in `config/config.yaml` presets:

```yaml
enhancement_presets:
  viral_shorts:
    branding:
      watermark:
        logo_path: "assets/logos/logo.png"
    captions:
      style:
        font_family: "assets/fonts/Montserrat-ExtraBold.ttf"
    mascot:
      assets_path: "assets/mascot/"
      variants: ["pop", "bounce", "spin"]
```

## 📋 Installation Notes

1. **Fonts**: Install system-wide or place in `assets/fonts/` 
2. **Missing Assets**: System gracefully handles missing files
3. **Fallbacks**: Default fonts/colors used when assets unavailable
4. **Performance**: Compress large assets to reduce processing time

## 🎯 Best Practices

- **Keep file sizes small** for faster processing
- **Use consistent naming** across asset variants  
- **Test all assets** with the enhancement system
- **Backup important assets** before making changes
- **Update config** when adding new asset types
