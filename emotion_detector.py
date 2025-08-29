#!/usr/bin/env python3
"""
AI-Powered Emotion Detection for Streamers
==========================================

Detects streamer emotions (hype, tilt, shock, etc.) using:
- Facial expression analysis
- Voice tone analysis  
- Gesture detection
- Chat sentiment correlation

Triggers different viral effects based on emotional state.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time
import librosa
import moviepy.editor as mp
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class EmotionFrame:
    """Single frame emotion analysis result"""
    timestamp: float
    emotion: str  # 'hype', 'tilt', 'shock', 'calm', 'confused'
    confidence: float
    face_detected: bool
    voice_intensity: float
    
@dataclass
class EmotionSegment:
    """Emotion segment for triggering effects"""
    start_time: float
    end_time: float
    dominant_emotion: str
    intensity: float  # 0.0 - 1.0
    trigger_effects: List[str]

class StreamerEmotionDetector:
    """AI-powered emotion detection for viral clip enhancement"""
    
    def __init__(self):
        """Initialize emotion detection system"""
        self.emotion_history: deque = deque(maxlen=300)  # 10 seconds at 30fps
        
        # Load pre-trained models
        self._load_face_detection()
        self._load_emotion_classifier()
        
        # Emotion thresholds and mappings (made more strict to reduce false positives)
        self.emotion_config = {
            'hype': {
                'audio_threshold': 0.8,  # Much louder required for hype
                'face_features': ['mouth_open', 'eyes_wide', 'eyebrows_raised'],
                'effects': ['zoom_in', 'shake', 'flash', 'particles', 'hype_text'],
                'duration_min': 1.5,  # Longer minimum duration
                'cooldown': 5.0       # Longer cooldown between detections
            },
            'tilt': {
                'audio_threshold': 0.9,  # Very strict for tilt detection
                'face_features': ['frown', 'eyes_narrow', 'eyebrows_down'],
                'effects': ['red_tint', 'angry_shake', 'tilt_text', 'slow_zoom'],
                'duration_min': 2.0,   # Longer minimum duration
                'cooldown': 8.0        # Much longer cooldown
            },
            'shock': {
                'audio_threshold': 0.6,  # Sudden loud reaction
                'face_features': ['mouth_open', 'eyes_wide', 'eyebrows_raised'],
                'effects': ['freeze_frame', 'zoom_in_fast', 'shock_text', 'flash'],
                'duration_min': 0.5,
                'cooldown': 2.0
            },
            'confused': {
                'audio_threshold': 0.3,  # Quieter, uncertain
                'face_features': ['eyebrows_raised', 'head_tilt'],
                'effects': ['question_marks', 'confused_text'],
                'duration_min': 1.0,
                'cooldown': 4.0
            }
        }
        
        # Text overlays for emotions
        self.emotion_texts = {
            'hype': ["LET'S GOOO! 🔥", "SHEESH! 😤", "ABSOLUTE MADNESS! 🚀", "INSANE! 💯"],
            'tilt': ["WHAT?! 😡", "NO WAY... 🤬", "RIGGED! 😤", "BRUHHH 💀"],
            'shock': ["NO SHOT! 😱", "WHAT THE?! 😳", "WAIT... 🤯", "HOLD UP! ⏸️"],
            'confused': ["HUH? 🤔", "WAIT WHAT? 😵", "I'M LOST... 🤷", "CONFUSED 🤨"]
        }
        
        # Last detection times for cooldowns
        self.last_emotion_time = {}
        
    def _load_face_detection(self):
        """Load OpenCV face detection model"""
        try:
            # Use OpenCV's DNN face detector (more accurate)
            self.face_net = cv2.dnn.readNetFromTensorflow(
                'assets/models/opencv_face_detector_uint8.pb',
                'assets/models/opencv_face_detector.pbtxt'
            )
            self.use_dnn_face = True
            logger.info("✅ Loaded DNN face detector")
        except:
            # Fallback to Haar cascades
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.use_dnn_face = False
            logger.info("⚠️ Using Haar cascade face detector (fallback)")
    
    def _load_emotion_classifier(self):
        """Load emotion classification model"""
        try:
            # Try to load a pre-trained emotion model
            import tensorflow as tf
            self.emotion_model = tf.keras.models.load_model('assets/models/emotion_model.h5')
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            self.use_ml_emotions = True
            logger.info("✅ Loaded ML emotion classifier")
        except:
            # Fallback to heuristic-based emotion detection
            self.use_ml_emotions = False
            logger.info("⚠️ Using heuristic emotion detection (fallback)")
    
    def detect_face_emotions(self, frame: np.ndarray, timestamp: float) -> Optional[EmotionFrame]:
        """Detect emotions from facial expressions in a frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self._detect_faces(gray)
            
            if len(faces) == 0:
                return EmotionFrame(timestamp, 'unknown', 0.0, False, 0.0)
            
            # Use the largest face (main streamer)
            face = max(faces, key=lambda x: x[2] * x[3])  # Sort by area
            x, y, w, h = face
            
            # Extract face region
            face_region = gray[y:y+h, x:x+w]
            
            if self.use_ml_emotions:
                emotion, confidence = self._classify_emotion_ml(face_region)
            else:
                emotion, confidence = self._classify_emotion_heuristic(frame, face)
            
            return EmotionFrame(timestamp, emotion, confidence, True, 0.0)
            
        except Exception as e:
            logger.warning(f"Face emotion detection failed: {e}")
            return EmotionFrame(timestamp, 'unknown', 0.0, False, 0.0)
    
    def _detect_faces(self, gray_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in grayscale frame"""
        if self.use_dnn_face:
            # DNN-based detection (more accurate)
            h, w = gray_frame.shape
            blob = cv2.dnn.blobFromImage(gray_frame, 1.0, (300, 300), [104, 117, 123])
            self.face_net.setInput(blob)
            detections = self.face_net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:  # Confidence threshold
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    faces.append((x1, y1, x2-x1, y2-y1))
            return faces
        else:
            # Haar cascade fallback
            faces = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            return [(x, y, w, h) for x, y, w, h in faces]
    
    def _classify_emotion_ml(self, face_region: np.ndarray) -> Tuple[str, float]:
        """Classify emotion using ML model"""
        try:
            # Preprocess face for model
            face_resized = cv2.resize(face_region, (48, 48))
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)
            face_input = np.expand_dims(face_input, axis=-1)
            
            # Predict emotion
            predictions = self.emotion_model.predict(face_input, verbose=0)
            emotion_idx = np.argmax(predictions)
            confidence = float(predictions[0][emotion_idx])
            
            # Map ML emotions to our emotion system
            ml_emotion = self.emotion_labels[emotion_idx]
            mapped_emotion = self._map_ml_to_streamer_emotion(ml_emotion, confidence)
            
            return mapped_emotion, confidence
            
        except Exception as e:
            logger.warning(f"ML emotion classification failed: {e}")
            return 'unknown', 0.0
    
    def _map_ml_to_streamer_emotion(self, ml_emotion: str, confidence: float) -> str:
        """Map ML emotion labels to streamer-specific emotions"""
        mapping = {
            'happy': 'hype' if confidence > 0.7 else 'calm',
            'surprise': 'shock',
            'angry': 'tilt',
            'fear': 'shock',
            'sad': 'tilt' if confidence > 0.6 else 'calm',
            'disgust': 'tilt',
            'neutral': 'calm'
        }
        return mapping.get(ml_emotion, 'calm')
    
    def _classify_emotion_heuristic(self, frame: np.ndarray, face: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """Classify emotion using heuristic analysis (fallback)"""
        try:
            x, y, w, h = face
            
            # Analyze mouth region (bottom 1/3 of face)
            mouth_region = frame[y + int(h*0.66):y + h, x:x + w]
            
            # Analyze eye region (top 1/3 of face)
            eye_region = frame[y:y + int(h*0.33), x:x + w]
            
            # Simple heuristics based on pixel intensity and movement
            mouth_intensity = np.mean(mouth_region) if mouth_region.size > 0 else 0
            eye_intensity = np.mean(eye_region) if eye_region.size > 0 else 0
            
            # Detect mouth opening (darker mouth region indicates open mouth)
            mouth_open = mouth_intensity < 100  # Dark mouth = open
            
            # Basic emotion classification
            if mouth_open:
                return 'hype', 0.6  # Open mouth suggests excitement/speaking
            else:
                return 'calm', 0.4
                
        except Exception as e:
            logger.warning(f"Heuristic emotion classification failed: {e}")
            return 'calm', 0.3
    
    def analyze_voice_intensity(self, audio_path: str, start_time: float, end_time: float) -> float:
        """Analyze voice intensity/volume for emotion detection"""
        try:
            # Load audio segment
            y, sr = librosa.load(audio_path, offset=start_time, duration=end_time-start_time)
            
            # Calculate RMS (volume/intensity)
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
            avg_intensity = float(np.mean(rms))
            
            # Normalize to 0-1 range
            normalized_intensity = min(avg_intensity * 10, 1.0)
            
            return normalized_intensity
            
        except Exception as e:
            logger.warning(f"Voice intensity analysis failed: {e}")
            return 0.0
    
    def detect_emotional_segments(self, video_path: str) -> List[EmotionSegment]:
        """Analyze entire video and detect emotional segments"""
        logger.info(f"🔍 Analyzing emotions in {video_path}")
        
        try:
            # Load video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            logger.info(f"📹 Video: {duration:.1f}s, {fps:.1f} FPS, {frame_count} frames")
            
            # Analyze frames at intervals (every 0.5 seconds)
            analysis_interval = 0.5
            frame_interval = int(fps * analysis_interval)
            
            emotion_frames = []
            frame_num = 0
            
            while frame_num < frame_count:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                timestamp = frame_num / fps
                emotion_frame = self.detect_face_emotions(frame, timestamp)
                
                if emotion_frame:
                    emotion_frames.append(emotion_frame)
                    
                    if len(emotion_frames) % 10 == 0:
                        logger.info(f"🎭 Analyzed {len(emotion_frames)} frames ({timestamp:.1f}s)")
                
                frame_num += frame_interval
            
            cap.release()
            
            # Analyze voice intensity
            self._add_voice_analysis(video_path, emotion_frames)
            
            # Group emotions into segments
            segments = self._group_emotion_segments(emotion_frames)
            
            logger.info(f"✅ Found {len(segments)} emotional segments")
            return segments
            
        except Exception as e:
            logger.error(f"❌ Emotion analysis failed: {e}")
            return []
    
    def _add_voice_analysis(self, video_path: str, emotion_frames: List[EmotionFrame]):
        """Add voice intensity analysis to emotion frames"""
        try:
            # Extract audio for analysis
            video = mp.VideoFileClip(video_path)
            if not video.audio:
                logger.warning("⚠️ No audio found for voice analysis")
                return
            
            temp_audio = "temp_emotion_audio.wav"
            video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
            
            # Analyze voice intensity for each frame
            for emotion_frame in emotion_frames:
                start_time = max(0, emotion_frame.timestamp - 0.25)  # 0.5s window
                end_time = min(video.duration, emotion_frame.timestamp + 0.25)
                
                voice_intensity = self.analyze_voice_intensity(temp_audio, start_time, end_time)
                emotion_frame.voice_intensity = voice_intensity
            
            # Cleanup
            video.close()
            Path(temp_audio).unlink(missing_ok=True)
            
        except Exception as e:
            logger.warning(f"Voice analysis failed: {e}")
    
    def _group_emotion_segments(self, emotion_frames: List[EmotionFrame]) -> List[EmotionSegment]:
        """Group emotion frames into coherent segments for effects"""
        if not emotion_frames:
            return []
        
        segments = []
        current_emotion = None
        segment_start = 0
        emotion_counts = {}
        
        for i, frame in enumerate(emotion_frames):
            # Skip unknown/undetected frames
            if frame.emotion == 'unknown' or not frame.face_detected:
                continue
            
            # Combine face emotion + voice intensity for final classification
            final_emotion = self._combine_emotion_signals(frame)
            
            # Start new segment if emotion changed significantly
            if current_emotion != final_emotion:
                # End previous segment
                if current_emotion and i > segment_start:
                    self._finalize_segment(segments, emotion_frames, segment_start, i-1, emotion_counts)
                
                # Start new segment
                current_emotion = final_emotion
                segment_start = i
                emotion_counts = {final_emotion: 1}
            else:
                # Continue current segment
                emotion_counts[final_emotion] = emotion_counts.get(final_emotion, 0) + 1
        
        # Finalize last segment
        if current_emotion and len(emotion_frames) > segment_start:
            self._finalize_segment(segments, emotion_frames, segment_start, len(emotion_frames)-1, emotion_counts)
        
        # Filter segments by minimum duration and confidence
        filtered_segments = []
        for segment in segments:
            min_duration = self.emotion_config.get(segment.dominant_emotion, {}).get('duration_min', 1.0)
            duration = segment.end_time - segment.start_time
            
            if duration >= min_duration and segment.intensity >= 0.7:  # Higher threshold
                # Check cooldown
                last_time = self.last_emotion_time.get(segment.dominant_emotion, 0)
                cooldown = self.emotion_config.get(segment.dominant_emotion, {}).get('cooldown', 3.0)
                
                if segment.start_time - last_time >= cooldown:
                    filtered_segments.append(segment)
                    self.last_emotion_time[segment.dominant_emotion] = segment.start_time
        
        return filtered_segments
    
    def _combine_emotion_signals(self, frame: EmotionFrame) -> str:
        """Combine facial emotion + voice intensity for final classification"""
        base_emotion = frame.emotion
        voice_intensity = frame.voice_intensity
        
        # Voice intensity can amplify or change emotion
        if voice_intensity > 0.8:  # Very loud
            if base_emotion in ['hype', 'shock']:
                return 'hype'  # Loud + positive = hype
            else:
                return 'tilt'  # Loud + negative = tilt
        elif voice_intensity > 0.6:  # Moderately loud
            if base_emotion == 'calm':
                return 'hype'  # Calm face but loud voice = hype
            return base_emotion
        else:
            # Low voice intensity
            if base_emotion == 'tilt' and voice_intensity < 0.3:
                return 'confused'  # Quiet tilt = confused
            return base_emotion
    
    def _finalize_segment(self, segments: List[EmotionSegment], frames: List[EmotionFrame], 
                         start_idx: int, end_idx: int, emotion_counts: Dict):
        """Create emotion segment from frame range"""
        if start_idx >= len(frames) or end_idx >= len(frames):
            return
        
        start_time = frames[start_idx].timestamp
        end_time = frames[end_idx].timestamp
        
        # Find dominant emotion
        dominant_emotion = max(emotion_counts.keys(), key=lambda x: emotion_counts[x])
        
        # Calculate intensity (confidence + voice intensity)
        total_confidence = sum(f.confidence for f in frames[start_idx:end_idx+1] if f.emotion == dominant_emotion)
        total_voice = sum(f.voice_intensity for f in frames[start_idx:end_idx+1])
        count = end_idx - start_idx + 1
        
        intensity = min((total_confidence + total_voice) / count, 1.0)
        
        # Get effects for this emotion
        effects = self.emotion_config.get(dominant_emotion, {}).get('effects', [])
        
        segment = EmotionSegment(
            start_time=start_time,
            end_time=end_time,
            dominant_emotion=dominant_emotion,
            intensity=intensity,
            trigger_effects=effects
        )
        
        segments.append(segment)
        logger.info(f"🎭 Emotion segment: {dominant_emotion} ({start_time:.1f}s-{end_time:.1f}s, intensity: {intensity:.2f})")
    
    def get_emotion_text_overlay(self, emotion: str, intensity: float) -> str:
        """Get text overlay for emotion"""
        texts = self.emotion_texts.get(emotion, ["STREAMER MOMENT! 🎮"])
        
        # Choose text based on intensity
        if intensity > 0.8:
            # High intensity - use more extreme text
            return np.random.choice(texts)
        else:
            # Lower intensity - use milder text
            return texts[0] if texts else "MOMENT! 🎮"


def test_emotion_detector():
    """Test the emotion detection system"""
    print("🤖 Testing Emotion Detection System")
    print("=" * 50)
    
    detector = StreamerEmotionDetector()
    
    # Test with a clip
    test_clip = "clips/test_clip.mp4"
    if Path(test_clip).exists():
        print(f"📹 Analyzing emotions in {test_clip}")
        
        segments = detector.detect_emotional_segments(test_clip)
        
        print(f"\n🎭 EMOTION ANALYSIS RESULTS:")
        print("-" * 30)
        
        for i, segment in enumerate(segments, 1):
            print(f"{i}. {segment.dominant_emotion.upper()}")
            print(f"   Time: {segment.start_time:.1f}s - {segment.end_time:.1f}s")
            print(f"   Intensity: {segment.intensity:.2f}")
            print(f"   Effects: {', '.join(segment.trigger_effects)}")
            print()
        
        if not segments:
            print("😐 No strong emotional segments detected")
            print("   Try with a more expressive clip!")
    else:
        print(f"❌ Test clip not found: {test_clip}")
        print("   Place a video file there to test!")


if __name__ == "__main__":
    test_emotion_detector()
