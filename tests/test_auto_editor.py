"""
Test Suite for Auto-Viral Editor v1
==================================

Tests the complete auto-viral editing pipeline with synthetic test assets.
"""

import unittest
import numpy as np
import tempfile
import os
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import moviepy.editor as mp
    from editing import audio, captions, effects, layout
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False


class TestAudioAnalysis(unittest.TestCase):
    """Test audio analysis functions."""
    
    def setUp(self):
        """Create test audio signal."""
        self.sr = 48000
        self.duration = 5.0  # 5 seconds
        self.samples = int(self.sr * self.duration)
        
        # Create synthetic audio with peaks at known times
        t = np.linspace(0, self.duration, self.samples)
        
        # Base signal
        signal = np.sin(2 * np.pi * 440 * t) * 0.1  # 440Hz tone at low volume
        
        # Add impact peaks at specific times
        self.peak_times = [1.0, 2.5, 4.0]
        for peak_time in self.peak_times:
            # Create burst at peak time
            peak_start = int(peak_time * self.sr)
            peak_end = int((peak_time + 0.1) * self.sr)
            if peak_end <= len(signal):
                signal[peak_start:peak_end] += np.sin(2 * np.pi * 1000 * 
                                                     t[peak_start:peak_end]) * 0.8
        
        self.audio_signal = signal
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modules not available")
    def test_rms_calculation(self):
        """Test RMS energy calculation."""
        rms_vals, hop = audio.rms(self.audio_signal, win_ms=50, sr=self.sr)
        
        # Should have reasonable number of frames
        expected_frames = int(self.duration * 1000 / 50)  # ~100 frames for 5s
        self.assertGreater(len(rms_vals), expected_frames * 0.8)
        self.assertLess(len(rms_vals), expected_frames * 1.2)
        
        # RMS values should be positive
        self.assertTrue(np.all(rms_vals >= 0))
        
        # Should have higher energy around peak times
        for peak_time in self.peak_times:
            peak_frame = int(peak_time / (hop / self.sr))
            if peak_frame < len(rms_vals):
                peak_rms = rms_vals[peak_frame]
                nearby_start = max(0, peak_frame - 5)
                nearby_end = min(len(rms_vals), peak_frame + 5)
                nearby_avg = np.mean(rms_vals[nearby_start:nearby_end])
                # Peak should be higher than surrounding average
                self.assertGreater(peak_rms, nearby_avg * 0.8)
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modules not available")
    def test_impact_detection(self):
        """Test impact peak detection."""
        detected_peaks, (rms_vals, z_scores, hop) = audio.detect_impacts(
            self.audio_signal, self.sr, z_thresh=1.5
        )
        
        # Should detect peaks near our synthetic peaks
        self.assertGreater(len(detected_peaks), 0)
        
        # Check that detected peaks are reasonably close to actual peaks
        for actual_peak in self.peak_times:
            distances = [abs(detected - actual_peak) for detected in detected_peaks]
            min_distance = min(distances) if distances else float('inf')
            self.assertLess(min_distance, 0.5, 
                           f"No detected peak within 0.5s of actual peak at {actual_peak}s")
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modules not available")
    def test_moment_timing(self):
        """Test moment timing calculation."""
        peak_times = [1.0, 2.5, 4.0]
        video_duration = 5.0
        
        moment = audio.get_moment_timing(peak_times, video_duration)
        
        self.assertIsNotNone(moment)
        self.assertIn('pre_start', moment)
        self.assertIn('impact', moment)
        self.assertIn('post_end', moment)
        
        # Timing should be reasonable
        self.assertGreaterEqual(moment['pre_start'], 0)
        self.assertLessEqual(moment['post_end'], video_duration)
        self.assertLess(moment['pre_start'], moment['impact'])
        self.assertLess(moment['impact'], moment['post_end'])


class TestSyntheticVideo(unittest.TestCase):
    """Test with synthetic video assets."""
    
    @classmethod
    def setUpClass(cls):
        """Create synthetic test video."""
        if not MODULES_AVAILABLE:
            return
        
        # Create a simple test video with color bars and audio burst
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_video_path = os.path.join(cls.temp_dir, "test_video.mp4")
        
        # Create video with color bars
        duration = 3.0
        fps = 30
        
        def make_frame(t):
            # Create color bars that change over time
            width, height = 640, 480
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Create vertical color bars
            bar_width = width // 6
            colors = [
                [255, 0, 0],    # Red
                [0, 255, 0],    # Green  
                [0, 0, 255],    # Blue
                [255, 255, 0],  # Yellow
                [255, 0, 255],  # Magenta
                [0, 255, 255]   # Cyan
            ]
            
            for i, color in enumerate(colors):
                start_x = i * bar_width
                end_x = min((i + 1) * bar_width, width)
                frame[:, start_x:end_x] = color
            
            # Add motion at peak times
            if 1.0 <= t <= 1.5:  # Motion during "peak"
                offset = int(20 * np.sin(t * 20))  # Fast shake
                frame = np.roll(frame, offset, axis=1)
            
            return frame
        
        # Create video clip
        video_clip = mp.VideoClip(make_frame, duration=duration)
        
        # Create audio with burst at 1.25s
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Base tone
        audio_signal = np.sin(2 * np.pi * 200 * t) * 0.1
        
        # Add burst at 1.25s
        burst_start = int(1.25 * sample_rate)
        burst_end = int(1.35 * sample_rate)
        audio_signal[burst_start:burst_end] += np.sin(2 * np.pi * 800 * t[burst_start:burst_end]) * 0.7
        
        # Create audio clip
        def make_audio_frame(t):
            if isinstance(t, np.ndarray):
                # Handle array of time values
                frames = []
                for time_val in t:
                    idx = int(time_val * sample_rate)
                    if idx < len(audio_signal):
                        frames.append([audio_signal[idx]])
                    else:
                        frames.append([0.0])
                return np.array(frames)
            else:
                # Handle single time value
                idx = int(t * sample_rate)
                if idx < len(audio_signal):
                    return np.array([[audio_signal[idx]]])
                else:
                    return np.array([[0.0]])
        
        audio_clip = mp.AudioClip(make_audio_frame, duration=duration)
        
        # Combine and save
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(
            cls.test_video_path,
            fps=fps,
            verbose=False,
            logger=None
        )
        final_clip.close()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            import shutil
            shutil.rmtree(cls.temp_dir)
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modules not available")
    def test_layout_conversion(self):
        """Test video layout conversion to TikTok format."""
        if not hasattr(self, 'test_video_path'):
            self.skipTest("Test video not available")
        
        clip = mp.VideoFileClip(self.test_video_path)
        
        # Test conversion to TikTok format
        tiktok_clip = layout.apply_tiktok_layout(clip, target_size=(1080, 1920))
        
        # Check dimensions
        self.assertEqual(tiktok_clip.w, 1080)
        self.assertEqual(tiktok_clip.h, 1920)
        
        # Check duration preserved
        self.assertAlmostEqual(tiktok_clip.duration, clip.duration, places=1)
        
        clip.close()
        tiktok_clip.close()
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modules not available")
    def test_effects_application(self):
        """Test visual effects application."""
        if not hasattr(self, 'test_video_path'):
            self.skipTest("Test video not available")
        
        clip = mp.VideoFileClip(self.test_video_path)
        peak_times = [1.25]  # Known peak from synthetic audio
        
        # Test micro-shake
        shaken_clip = effects.apply_micro_shake(clip, peak_times, px=3, dur=0.1)
        self.assertEqual(shaken_clip.duration, clip.duration)
        
        # Test impact flash
        flash_clip = effects.apply_impact_flash(clip, peak_times, dur_frames=2)
        self.assertEqual(flash_clip.duration, clip.duration)
        
        # Test punch zoom
        zoom_clip = effects.apply_punch_zoom(clip, peak_times, max_scale=1.02)
        self.assertEqual(zoom_clip.duration, clip.duration)
        
        clip.close()
        shaken_clip.close()
        flash_clip.close()
        zoom_clip.close()
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modules not available")
    def test_hook_overlay(self):
        """Test hook overlay placement."""
        if not hasattr(self, 'test_video_path'):
            self.skipTest("Test video not available")
        
        clip = mp.VideoFileClip(self.test_video_path)
        hook_text = "TEST HOOK OVERLAY"
        
        hook_clip = effects.add_hook_overlay(
            clip, hook_text, duration=0.8, video_size=(clip.w, clip.h)
        )
        
        # Should have same duration
        self.assertEqual(hook_clip.duration, clip.duration)
        
        # Should be a composite clip
        self.assertIsInstance(hook_clip, mp.CompositeVideoClip)
        
        clip.close()
        hook_clip.close()
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modules not available")
    def test_loop_ending(self):
        """Test loop ending addition."""
        if not hasattr(self, 'test_video_path'):
            self.skipTest("Test video not available")
        
        clip = mp.VideoFileClip(self.test_video_path)
        original_duration = clip.duration
        
        loop_clip = effects.add_loop_ending(clip, loop_frames=10, fps=30)
        
        # Duration should increase slightly
        self.assertGreater(loop_clip.duration, original_duration)
        self.assertLess(loop_clip.duration, original_duration + 1.0)  # Less than 1s added
        
        clip.close()
        loop_clip.close()


class TestMockTranscription(unittest.TestCase):
    """Test caption system with mock transcription."""
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modules not available")
    def test_chunk_splitting(self):
        """Test transcript chunking for captions."""
        # Mock word list with timestamps
        mock_words = [
            {'word': 'this', 'start': 0.0, 'end': 0.3},
            {'word': 'is', 'start': 0.3, 'end': 0.5},
            {'word': 'a', 'start': 0.5, 'end': 0.6},
            {'word': 'test', 'start': 0.6, 'end': 1.0},
            {'word': 'of', 'start': 1.0, 'end': 1.2},
            {'word': 'caption', 'start': 1.2, 'end': 1.8},
            {'word': 'splitting', 'start': 1.8, 'end': 2.5},
            {'word': 'functionality', 'start': 2.5, 'end': 3.5},
        ]
        
        chunks = captions.split_into_chunks(mock_words, max_chars_per_line=12, max_lines=2)
        
        # Should create reasonable chunks
        self.assertGreater(len(chunks), 0)
        
        for chunk in chunks:
            # Check required fields
            self.assertIn('text', chunk)
            self.assertIn('start', chunk)
            self.assertIn('end', chunk)
            self.assertIn('words', chunk)
            
            # Check timing
            self.assertLessEqual(chunk['start'], chunk['end'])
            
            # Check line count (max 2 lines)
            lines = chunk['text'].split('\n')
            self.assertLessEqual(len(lines), 2)
            
            # Check line length
            for line in lines:
                self.assertLessEqual(len(line), 12)
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modules not available")
    def test_emphasis_detection(self):
        """Test emphasis word detection."""
        test_cases = [
            ("this is INSANE", ['insane']),
            ("WOW that was amazing!", ['wow', 'amazing']),
            ("no way this is CRAZY!!!", ['crazy']),
            ("regular text here", []),
        ]
        
        for text, expected_words in test_cases:
            detected = captions.identify_emphasis_words(text)
            
            for expected in expected_words:
                self.assertIn(expected, detected, 
                             f"Expected '{expected}' in emphasis words for '{text}'")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modules not available")
    def test_breakdown_parsing(self):
        """Test breakdown JSON parsing."""
        # Mock breakdown data
        breakdown = {
            "z": {"kw": 2.5, "sent": 1.8},
            "viewer_delta_percent": 15,
            "current_mps": 4.2,
            "components": {
                "keyword": ["insane", "clutch"],
                "sentiment": 0.8
            }
        }
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(breakdown, f)
            breakdown_path = f.name
        
        try:
            # Test hook generation
            from clip_enhancer_v2 import hook_from_breakdown
            
            hook_text = hook_from_breakdown(breakdown, "sample transcript text")
            
            # Should generate appropriate hook based on high keyword score
            self.assertEqual(hook_text, "WAIT FOR IT… INSANE")
            
        finally:
            os.unlink(breakdown_path)
    
    @unittest.skipUnless(MODULES_AVAILABLE, "Modules not available")
    def test_safe_area_validation(self):
        """Test TikTok safe area validation."""
        video_size = (1080, 1920)
        
        # Test safe positions
        safe_cases = [
            ((540, 300), (200, 100)),  # Center, small element
            ((100, 400), (300, 200)),  # Left side, medium element
        ]
        
        for position, size in safe_cases:
            is_safe = layout.validate_safe_areas(position, size, video_size)
            self.assertTrue(is_safe, f"Position {position} size {size} should be safe")
        
        # Test unsafe positions
        unsafe_cases = [
            ((540, 50), (200, 100)),   # Too close to top UI
            ((540, 1800), (200, 100)), # Too close to bottom UI
            ((950, 400), (200, 100)),   # Too close to right UI
        ]
        
        for position, size in unsafe_cases:
            is_safe = layout.validate_safe_areas(position, size, video_size)
            self.assertFalse(is_safe, f"Position {position} size {size} should be unsafe")


def create_test_suite():
    """Create a test suite with all test cases."""
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAudioAnalysis,
        TestSyntheticVideo,
        TestMockTranscription,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_tests():
    """Run all tests and return success status."""
    if not MODULES_AVAILABLE:
        print("❌ Required modules not available for testing")
        print("   Install with: pip install moviepy openai-whisper opencv-python")
        return False
    
    print("🧪 Running Auto-Viral Editor test suite...")
    
    # Create and run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    
    print(f"\n{'='*50}")
    print(f"Test Results Summary:")
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {total_tests - failures - errors}")
    print(f"  Failed: {failures}")
    print(f"  Errors: {errors}")
    print(f"  Skipped: {skipped}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
