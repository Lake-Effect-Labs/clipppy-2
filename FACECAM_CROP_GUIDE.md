# Facecam Crop Configuration Guide

## Problem
Different streamers have their facecams in different positions:
- **theburntpeanut**: Bottom-left corner
- **timthetatman**: Bottom-right corner  
- **cloakzy**: Bottom-left corner
- **nickmercs**: Bottom-right corner
- **hutchmf**: (needs to be determined)

The automatic face detection doesn't work reliably for all streamers, so we need manual configuration.

## Solution
Added per-streamer `crop_regions` configuration in `config/config.yaml` that allows you to manually specify where the facecam is located.

## How to Configure

### Step 1: Find the Facecam Position
Watch a clip or stream and note where the facecam is positioned:
- **Bottom-left**: `[0.02, 0.70, 0.30, 0.98]`
- **Bottom-right**: `[0.70, 0.70, 0.98, 0.98]`
- **Top-left**: `[0.02, 0.02, 0.30, 0.30]`
- **Top-right**: `[0.70, 0.02, 0.98, 0.30]`

The format is: `[x1, y1, x2, y2]` where values are percentages (0.0 to 1.0):
- `x1, y1`: Top-left corner of facecam region
- `x2, y2`: Bottom-right corner of facecam region

### Step 2: Add to Config
In `config/config.yaml`, under each streamer's `enhancement.overrides`, add:

```yaml
enhancement:
  preset: viral_shorts
  overrides:
    crop_regions:
      # Facecam location (x1, y1, x2, y2 as percentages 0-1)
      face: [0.70, 0.70, 0.98, 0.98]  # bottom-right corner
      gameplay: [0.0, 0.0, 1.0, 1.0]  # full screen
    captions:
      enabled: true
      # ... rest of config
```

### Step 3: Test and Adjust
1. Run the enhancement on a test clip
2. Check if the facecam circle is positioned correctly
3. Adjust the values if needed:
   - Move **left**: decrease x1 and x2
   - Move **right**: increase x1 and x2
   - Move **up**: decrease y1 and y2
   - Move **down**: increase y1 and y2
   - Make **bigger**: increase the difference between x1-x2 and y1-y2
   - Make **smaller**: decrease the difference between x1-x2 and y1-y2

## Current Configuration

### timthetatman
```yaml
crop_regions:
  face: [0.75, 0.65, 0.98, 0.95]  # bottom-right corner
  gameplay: [0.0, 0.0, 1.0, 1.0]  # full screen
```

### cloakzy
```yaml
crop_regions:
  face: [0.02, 0.70, 0.30, 0.98]  # bottom-left corner
  gameplay: [0.0, 0.0, 1.0, 1.0]  # full screen
```

### nickmercs
```yaml
crop_regions:
  face: [0.75, 0.15, 0.98, 0.45]  # TOP-RIGHT corner (not bottom!)
  gameplay: [0.0, 0.0, 1.0, 1.0]  # full screen
```

### theburntpeanut
```yaml
crop_regions:
  face: [0.12, 0.55, 0.42, 0.95]  # bottom-left corner (custom position)
  gameplay: [0.0, 0.0, 1.0, 1.0]  # full screen
```

### hutchmf
**TODO**: Needs to be configured based on stream observation.

## How It Works

1. **Config Priority**: Manual `crop_regions` in config take priority over automatic detection
2. **Circular Overlay**: The facecam is cropped from the specified region, resized to a circle, and overlaid at the top-center of the video
3. **Fallback**: If no manual regions are specified, the system falls back to automatic detection (which may not work well)

## Benefits

✅ **Consistent Results**: Same facecam position every time  
✅ **No Detection Failures**: Doesn't rely on face detection AI  
✅ **Per-Streamer Customization**: Each streamer can have their own layout  
✅ **Easy to Adjust**: Simple percentage-based coordinates  

## Troubleshooting

### Facecam is cut off
- Expand the crop region by adjusting the coordinates
- Example: `[0.65, 0.65, 0.98, 0.98]` → `[0.60, 0.60, 0.98, 0.98]`

### Facecam includes too much gameplay
- Shrink the crop region
- Example: `[0.70, 0.70, 0.98, 0.98]` → `[0.75, 0.75, 0.98, 0.98]`

### Facecam is in wrong position
- Check the logs for "Manual facecam region" to see what coordinates are being used
- Adjust the x1, y1, x2, y2 values accordingly
- Remember: 0.0 = left/top edge, 1.0 = right/bottom edge

### No facecam overlay appears
- Check if `crop_regions` is properly indented in the YAML
- Ensure the values are in the correct format: `[x1, y1, x2, y2]`
- Check logs for "Using MANUAL crop regions from config"

## Next Steps

1. **Test with hutchmf**: Watch a clip and determine facecam position
2. **Fine-tune existing configs**: Test clips from each streamer and adjust if needed
3. **Document positions**: Keep track of which streamers use which layouts

