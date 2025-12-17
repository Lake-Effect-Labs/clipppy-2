"""
Layout Module - Smart cropping, safe areas, and TikTok formatting
===============================================================

Handles video layout conversion from any aspect ratio to TikTok-optimized 9:16.
"""

import cv2
import numpy as np
import moviepy.editor as mp
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# TikTok safe area definitions
TIKTOK_SAFE_AREAS = {
    'top_ui': 0.12,      # Top 12% reserved for profile/effects UI
    'bottom_ui': 0.12,   # Bottom 12% reserved for interaction UI
    'right_ui': 0.08,    # Right 8% reserved for buttons
    'content_top': 0.15, # Start content at 15%
    'content_bottom': 0.85  # End content at 85%
}

TARGET_SIZE = (1080, 1920)  # TikTok standard resolution


def detect_face_regions(
    video_path: str,
    sample_frames: int = 10,
    confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    Detect face regions in video for smart cropping.
    
    Args:
        video_path: Path to video file
        sample_frames: Number of frames to sample for analysis
        confidence_threshold: Minimum confidence for face detection
        
    Returns:
        List of face region dictionaries with coordinates and confidence
    """
    try:
        # Load OpenCV DNN face detector
        net = cv2.dnn.readNetFromTensorflow(
            'assets/models/opencv_face_detector_uint8.pb',
            'assets/models/opencv_face_detector.pbtxt'
        )
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            logger.warning("Could not read video frames for face detection")
            return []
        
        face_regions = []
        frame_indices = np.linspace(0, total_frames - 1, sample_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            h, w = frame.shape[:2]
            
            # Create blob for DNN
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
            net.setInput(blob)
            detections = net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > confidence_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    face_regions.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'confidence': float(confidence),
                        'frame': frame_idx,
                        'center_x': (x1 + x2) // 2,
                        'center_y': (y1 + y2) // 2,
                        'width': x2 - x1,
                        'height': y2 - y1
                    })
        
        cap.release()
        
        logger.info(f"Detected {len(face_regions)} face regions across {sample_frames} frames")
        return face_regions
        
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return []


def analyze_motion_regions(
    video_path: str,
    sample_frames: int = 15,
    threshold: float = 30.0
) -> Dict:
    """
    Analyze motion patterns to identify important regions.
    
    Args:
        video_path: Path to video file
        sample_frames: Number of frames to analyze
        threshold: Motion threshold for detection
        
    Returns:
        Dictionary with motion statistics and hotspots
    """
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < 2:
            return {'motion_center': (0.5, 0.5), 'intensity': 0.0}
        
        frame_indices = np.linspace(0, total_frames - 2, sample_frames, dtype=int)
        motion_accumulator = None
        prev_gray = None
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret1, frame1 = cap.read()
            ret2, frame2 = cap.read()
            
            if not (ret1 and ret2):
                continue
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                gray1, gray2,
                np.array([[]], dtype=np.float32),
                None
            )
            
            # Calculate frame difference for motion intensity
            diff = cv2.absdiff(gray1, gray2)
            motion_mask = (diff > threshold).astype(np.uint8)
            
            if motion_accumulator is None:
                motion_accumulator = motion_mask.astype(np.float32)
            else:
                motion_accumulator += motion_mask.astype(np.float32)
        
        cap.release()
        
        if motion_accumulator is not None:
            # Find motion center of mass
            moments = cv2.moments(motion_accumulator)
            if moments['m00'] > 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
                
                h, w = motion_accumulator.shape
                motion_center = (cx / w, cy / h)  # Normalized coordinates
                intensity = np.sum(motion_accumulator) / (w * h * sample_frames)
            else:
                motion_center = (0.5, 0.5)
                intensity = 0.0
        else:
            motion_center = (0.5, 0.5)
            intensity = 0.0
        
        return {
            'motion_center': motion_center,
            'intensity': float(intensity),
            'has_motion': intensity > 5.0
        }
        
    except Exception as e:
        logger.error(f"Motion analysis failed: {e}")
        return {'motion_center': (0.5, 0.5), 'intensity': 0.0}


def calculate_smart_crop(
    video_size: Tuple[int, int],
    target_size: Tuple[int, int] = TARGET_SIZE,
    face_regions: List[Dict] = None,
    motion_data: Dict = None,
    content_type: str = 'auto'
) -> Dict:
    """
    Calculate optimal crop region for TikTok format.
    
    Args:
        video_size: Original video dimensions (width, height)
        target_size: Target dimensions (width, height)
        face_regions: List of detected face regions
        motion_data: Motion analysis data
        content_type: 'facecam', 'gameplay', 'auto'
        
    Returns:
        Dictionary with crop parameters
    """
    orig_w, orig_h = video_size
    target_w, target_h = target_size
    target_aspect = target_w / target_h
    orig_aspect = orig_w / orig_h
    
    crop_info = {
        'x': 0, 'y': 0, 'width': orig_w, 'height': orig_h,
        'scale': 1.0, 'method': 'center_crop'
    }
    
    if orig_aspect > target_aspect:
        # Source is wider - need to crop horizontally
        new_width = int(orig_h * target_aspect)
        x_offset = (orig_w - new_width) // 2
        
        # Adjust based on face/motion data
        if face_regions:
            # Average face center X
            face_centers_x = [f['center_x'] for f in face_regions]
            avg_face_x = sum(face_centers_x) / len(face_centers_x)
            
            # Shift crop to include faces
            face_offset = avg_face_x - orig_w // 2
            x_offset = max(0, min(x_offset + int(face_offset * 0.7), orig_w - new_width))
            crop_info['method'] = 'face_aware'
            
        elif motion_data and motion_data.get('has_motion'):
            # Shift crop toward motion center
            motion_x = motion_data['motion_center'][0] * orig_w
            motion_offset = motion_x - orig_w // 2
            x_offset = max(0, min(x_offset + int(motion_offset * 0.5), orig_w - new_width))
            crop_info['method'] = 'motion_aware'
        
        crop_info.update({
            'x': x_offset,
            'y': 0,
            'width': new_width,
            'height': orig_h
        })
        
    elif orig_aspect < target_aspect:
        # Source is taller - need to crop vertically
        new_height = int(orig_w / target_aspect)
        y_offset = (orig_h - new_height) // 2
        
        # For vertical crops, prefer upper portion for face content
        if face_regions:
            face_centers_y = [f['center_y'] for f in face_regions]
            avg_face_y = sum(face_centers_y) / len(face_centers_y)
            
            # Shift crop to include faces, but keep some headroom
            face_offset = avg_face_y - orig_h // 2
            y_offset = max(0, min(y_offset + int(face_offset * 0.5), orig_h - new_height))
            crop_info['method'] = 'face_aware'
            
        elif content_type == 'gameplay':
            # For gameplay, prefer center or slightly lower
            y_offset = int(orig_h * 0.1)  # Start 10% from top
            crop_info['method'] = 'gameplay_optimized'
        
        crop_info.update({
            'x': 0,
            'y': y_offset,
            'width': orig_w,
            'height': new_height
        })
    
    # Calculate scaling factor to reach target size
    crop_info['scale'] = target_w / crop_info['width']
    
    logger.info(f"Smart crop: {crop_info['method']} - "
                f"{crop_info['width']}x{crop_info['height']} "
                f"at ({crop_info['x']}, {crop_info['y']}) "
                f"scale {crop_info['scale']:.2f}")
    
    return crop_info


def apply_tiktok_layout(
    clip: mp.VideoClip,
    video_path: Optional[str] = None,
    target_size: Tuple[int, int] = TARGET_SIZE,
    content_type: str = 'auto',
    safe_areas: bool = True
) -> mp.VideoClip:
    """
    Convert video to TikTok-optimized 9:16 layout.
    
    Args:
        clip: Source video clip
        video_path: Path to video file for analysis (optional)
        target_size: Target dimensions
        content_type: Content type hint for cropping
        safe_areas: Whether to respect TikTok safe areas
        
    Returns:
        Formatted video clip
    """
    orig_size = (clip.w, clip.h)
    target_w, target_h = target_size
    
    # Analyze video for smart cropping if path provided
    face_regions = []
    motion_data = None
    
    if video_path:
        try:
            face_regions = detect_face_regions(video_path)
            motion_data = analyze_motion_regions(video_path)
        except Exception as e:
            logger.warning(f"Video analysis failed, using center crop: {e}")
    
    # Calculate optimal crop
    crop_info = calculate_smart_crop(
        orig_size, target_size, face_regions, motion_data, content_type
    )
    
    # Apply crop and resize
    try:
        if crop_info['width'] != orig_size[0] or crop_info['height'] != orig_size[1]:
            # Need to crop
            cropped = clip.crop(
                x1=crop_info['x'],
                y1=crop_info['y'],
                x2=crop_info['x'] + crop_info['width'],
                y2=crop_info['y'] + crop_info['height']
            )
        else:
            cropped = clip
        
        # Resize to target dimensions
        formatted = cropped.resize((target_w, target_h))
        
        # Add letterboxing if needed to maintain safe areas
        if safe_areas and (target_h > target_w * 2):  # Very tall aspect ratio
            # Add subtle letterboxing to reduce effective height
            letterbox_height = int(target_h * 0.05)  # 5% letterbox
            
            letterbox_top = mp.ColorClip(
                size=(target_w, letterbox_height),
                color=(0, 0, 0)
            ).set_duration(formatted.duration).set_opacity(0.8)
            
            letterbox_bottom = mp.ColorClip(
                size=(target_w, letterbox_height),
                color=(0, 0, 0)
            ).set_duration(formatted.duration).set_opacity(0.8)
            
            letterbox_top = letterbox_top.set_position(('center', 0))
            letterbox_bottom = letterbox_bottom.set_position(('center', target_h - letterbox_height))
            
            formatted = mp.CompositeVideoClip([formatted, letterbox_top, letterbox_bottom])
        
        logger.info(f"Applied TikTok layout: {orig_size} → {target_size} using {crop_info['method']}")
        return formatted
        
    except Exception as e:
        logger.error(f"Layout formatting failed: {e}")
        # Fallback to simple resize
        return clip.resize((target_w, target_h))


def get_safe_position(
    area: str,
    video_size: Tuple[int, int] = TARGET_SIZE
) -> Tuple[str, int]:
    """
    Get safe positioning coordinates for TikTok UI elements.
    
    Args:
        area: 'top', 'bottom', 'center'
        video_size: Video dimensions
        
    Returns:
        Tuple of (horizontal_align, vertical_position)
    """
    width, height = video_size
    
    if area == 'top':
        # Position in top safe area
        y_pos = int(height * TIKTOK_SAFE_AREAS['content_top'])
        return ('center', y_pos)
        
    elif area == 'bottom':
        # Position in bottom safe area
        y_pos = int(height * (1 - TIKTOK_SAFE_AREAS['bottom_ui'] - 0.05))
        return ('center', y_pos)
        
    else:  # center
        # Position in middle safe area
        y_pos = int(height * 0.5)
        return ('center', y_pos)


def validate_safe_areas(
    clip_position: Tuple,
    clip_size: Tuple[int, int],
    video_size: Tuple[int, int] = TARGET_SIZE
) -> bool:
    """
    Validate that a clip doesn't overlap TikTok UI areas.
    
    Args:
        clip_position: (x, y) position of clip
        clip_size: (width, height) of clip
        video_size: Video dimensions
        
    Returns:
        True if position is safe, False if overlaps UI
    """
    x, y = clip_position
    clip_w, clip_h = clip_size
    video_w, video_h = video_size
    
    # Convert to normalized coordinates
    norm_x = x / video_w
    norm_y = y / video_h
    norm_w = clip_w / video_w
    norm_h = clip_h / video_h
    
    # Check top UI overlap
    if norm_y < TIKTOK_SAFE_AREAS['top_ui']:
        return False
    
    # Check bottom UI overlap
    if (norm_y + norm_h) > (1 - TIKTOK_SAFE_AREAS['bottom_ui']):
        return False
    
    # Check right UI overlap
    if (norm_x + norm_w) > (1 - TIKTOK_SAFE_AREAS['right_ui']):
        return False
    
    return True

