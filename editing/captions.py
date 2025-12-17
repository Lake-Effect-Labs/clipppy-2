"""
Caption Generation Module - Whisper transcription with viral styling
==================================================================

Provides dynamic, TikTok-optimized captions with per-word emphasis and safe areas.
"""

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import re
import moviepy.editor as mp

logger = logging.getLogger(__name__)

# TikTok-optimized caption settings
CAPTION_STYLE = {
    'fontsize': 48,
    'font': 'Arial-Bold',
    'color': 'white',
    'stroke_color': 'black',
    'stroke_width': 3,
    'method': 'caption',
    'align': 'center',
    'kerning': -2
}

EMPHASIS_WORDS = {
    'insane', 'crazy', 'unbelievable', 'amazing', 'wow', 'omg', 'wtf', 'no way',
    'clutch', 'perfect', 'impossible', 'legendary', 'godlike', 'nasty', 'sick'
}

SAFE_AREAS = {
    'top': {'y_min': 0.12, 'y_max': 0.55},      # Above TikTok profile/effects
    'bottom': {'y_min': 0.65, 'y_max': 0.88}    # Above TikTok UI elements
}


def transcribe_with_timestamps(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None
) -> Dict:
    """
    Transcribe audio using Whisper with word-level timestamps.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Target language (None for auto-detection)
        
    Returns:
        Whisper transcription result with segments and word timestamps
    """
    try:
        model = whisper.load_model(model_size)
        logger.info(f"Transcribing with Whisper {model_size} model...")
        
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            language=language,
            verbose=False
        )
        
        logger.info(f"Transcription complete: {len(result.get('segments', []))} segments")
        return result
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {'segments': [], 'text': ''}


def split_into_chunks(
    words: List[Dict],
    max_chars_per_line: int = 12,
    max_lines: int = 2,
    min_duration: float = 0.8,
    max_duration: float = 4.0
) -> List[Dict]:
    """
    Split word list into 2-line caption chunks with timing.
    
    Args:
        words: List of word dictionaries with 'word', 'start', 'end'
        max_chars_per_line: Maximum characters per line
        max_lines: Maximum lines per caption
        min_duration: Minimum chunk duration in seconds
        max_duration: Maximum chunk duration in seconds
        
    Returns:
        List of caption chunks with 'text', 'start', 'end', 'words'
    """
    if not words:
        return []
    
    chunks = []
    current_chunk = []
    current_line = ""
    current_lines = []
    
    for word in words:
        word_text = word.get('word', '').strip()
        if not word_text:
            continue
            
        # Check if adding word exceeds line length
        test_line = f"{current_line} {word_text}".strip()
        
        if len(test_line) <= max_chars_per_line:
            # Word fits on current line
            current_line = test_line
            current_chunk.append(word)
        else:
            # Need new line or new chunk
            if current_line:
                current_lines.append(current_line.upper())
                current_line = word_text
                
                if len(current_lines) >= max_lines:
                    # Complete current chunk
                    if current_chunk:
                        chunk_text = '\n'.join(current_lines)
                        chunk_start = current_chunk[0].get('start', 0)
                        chunk_end = current_chunk[-1].get('end', chunk_start + 1)
                        duration = chunk_end - chunk_start
                        
                        # Ensure minimum duration
                        if duration < min_duration:
                            chunk_end = chunk_start + min_duration
                            
                        chunks.append({
                            'text': chunk_text,
                            'start': chunk_start,
                            'end': chunk_end,
                            'words': current_chunk.copy(),
                            'duration': chunk_end - chunk_start
                        })
                    
                    # Start new chunk
                    current_chunk = [word]
                    current_lines = [word_text.upper()]
                    current_line = ""
                else:
                    current_chunk.append(word)
            else:
                current_line = word_text
                current_chunk.append(word)
    
    # Handle remaining content
    if current_line:
        current_lines.append(current_line.upper())
    
    if current_chunk and current_lines:
        chunk_text = '\n'.join(current_lines)
        chunk_start = current_chunk[0].get('start', 0)
        chunk_end = current_chunk[-1].get('end', chunk_start + 1)
        duration = chunk_end - chunk_start
        
        if duration < min_duration:
            chunk_end = chunk_start + min_duration
            
        chunks.append({
            'text': chunk_text,
            'start': chunk_start,
            'end': chunk_end,
            'words': current_chunk.copy(),
            'duration': chunk_end - chunk_start
        })
    
    logger.info(f"Split transcript into {len(chunks)} caption chunks")
    return chunks


def identify_emphasis_words(text: str) -> List[str]:
    """
    Identify words that should be emphasized in captions.
    
    Args:
        text: Caption text
        
    Returns:
        List of words to emphasize
    """
    words_to_emphasize = []
    text_lower = text.lower()
    
    # Check against emphasis word list
    for word in EMPHASIS_WORDS:
        if word in text_lower:
            words_to_emphasize.append(word)
    
    # Check for ALL CAPS words (already emphasized in original)
    caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
    words_to_emphasize.extend([w.lower() for w in caps_words])
    
    # Check for punctuation emphasis
    exclamation_words = re.findall(r'\b(\w+)!+', text, re.IGNORECASE)
    words_to_emphasize.extend([w.lower() for w in exclamation_words])
    
    return list(set(words_to_emphasize))


def create_caption_clip(
    chunk: Dict,
    video_size: Tuple[int, int] = (1080, 1920),
    safe_area: str = 'bottom',
    emphasis_scale: float = 1.15
) -> mp.TextClip:
    """
    Create a styled caption clip for a text chunk.
    
    Args:
        chunk: Caption chunk with 'text', 'start', 'end'
        video_size: Target video dimensions (width, height)
        safe_area: 'top' or 'bottom' safe area placement
        emphasis_scale: Scale factor for emphasized words
        
    Returns:
        MoviePy TextClip with styling and positioning
    """
    text = chunk['text']
    duration = chunk['end'] - chunk['start']
    
    # Identify words to emphasize
    emphasis_words = identify_emphasis_words(text)
    
    # Create base text clip with viral styling
    txt_clip = mp.TextClip(
        text,
        fontsize=CAPTION_STYLE['fontsize'],
        font=CAPTION_STYLE['font'],
        color=CAPTION_STYLE['color'],
        stroke_color=CAPTION_STYLE['stroke_color'],
        stroke_width=CAPTION_STYLE['stroke_width'],
        method=CAPTION_STYLE['method'],
        align=CAPTION_STYLE['align'],
        kerning=CAPTION_STYLE['kerning'],
        size=(video_size[0] * 0.9, None)  # 90% of video width
    ).set_duration(duration)
    
    # Position in safe area
    if safe_area == 'top':
        y_center = (SAFE_AREAS['top']['y_min'] + SAFE_AREAS['top']['y_max']) / 2
    else:
        y_center = (SAFE_AREAS['bottom']['y_min'] + SAFE_AREAS['bottom']['y_max']) / 2
    
    y_pos = int(video_size[1] * y_center)
    
    txt_clip = txt_clip.set_position(('center', y_pos))
    
    # Add subtle glow effect for viral look
    glow_clip = mp.TextClip(
        text,
        fontsize=CAPTION_STYLE['fontsize'] + 4,
        font=CAPTION_STYLE['font'],
        color='cyan',
        stroke_color='transparent',
        stroke_width=0,
        method=CAPTION_STYLE['method'],
        align=CAPTION_STYLE['align'],
        size=(video_size[0] * 0.9, None)
    ).set_duration(duration).set_position(('center', y_pos)).set_opacity(0.3)
    
    # Combine text with glow
    final_clip = mp.CompositeVideoClip([glow_clip, txt_clip])
    
    # Add emphasis animation if needed
    if emphasis_words and any(word.lower() in text.lower() for word in emphasis_words):
        # Simple scale pulse for emphasis
        def emphasis_resize(t):
            pulse = 1 + 0.1 * np.sin(t * 8)  # 4Hz pulse
            return pulse if any(word.lower() in text.lower() for word in emphasis_words) else 1.0
        
        final_clip = final_clip.resize(emphasis_resize)
    
    return final_clip


def generate_caption_sequence(
    transcript_result: Dict,
    video_duration: float,
    video_size: Tuple[int, int] = (1080, 1920)
) -> List[mp.TextClip]:
    """
    Generate complete caption sequence from transcript.
    
    Args:
        transcript_result: Whisper transcription result
        video_duration: Duration of source video
        video_size: Target video dimensions
        
    Returns:
        List of positioned and styled caption clips
    """
    caption_clips = []
    
    # Extract all words with timestamps
    all_words = []
    for segment in transcript_result.get('segments', []):
        for word_info in segment.get('words', []):
            all_words.append(word_info)
    
    if not all_words:
        logger.warning("No words with timestamps found in transcript")
        return []
    
    # Split into caption chunks
    chunks = split_into_chunks(all_words)
    
    # Create clips for each chunk
    for i, chunk in enumerate(chunks):
        # Alternate between top and bottom placement
        safe_area = 'bottom' if i % 2 == 0 else 'top'
        
        try:
            clip = create_caption_clip(chunk, video_size, safe_area)
            clip = clip.set_start(chunk['start'])
            caption_clips.append(clip)
            
        except Exception as e:
            logger.error(f"Failed to create caption clip {i}: {e}")
            continue
    
    logger.info(f"Generated {len(caption_clips)} caption clips")
    return caption_clips


def get_transcript_head(transcript_result: Dict, max_length: int = 28) -> str:
    """
    Extract first sentence from transcript for hook generation.
    
    Args:
        transcript_result: Whisper transcription result
        max_length: Maximum characters to return
        
    Returns:
        First sentence or phrase from transcript
    """
    text = transcript_result.get('text', '').strip()
    if not text:
        return ""
    
    # Split by sentence endings
    sentences = re.split(r'[.!?]+', text)
    if sentences:
        first_sentence = sentences[0].strip()
        if len(first_sentence) <= max_length:
            return first_sentence
        else:
            # Truncate to word boundary
            words = first_sentence.split()
            result = ""
            for word in words:
                test = f"{result} {word}".strip()
                if len(test) <= max_length:
                    result = test
                else:
                    break
            return result
    
    return text[:max_length].strip()
