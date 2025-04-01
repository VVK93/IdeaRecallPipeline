import re
from typing import Tuple, Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import logging

logger = logging.getLogger(__name__)

def is_valid_youtube_url(url: str) -> bool:
    """
    Validates if the provided URL is a valid YouTube URL.
    Supports various YouTube URL formats including:
    - Regular watch URLs (youtube.com/watch?v=...)
    - Short URLs (youtu.be/...)
    - Channel URLs (youtube.com/channel/...)
    - Playlist URLs (youtube.com/playlist?list=...)
    """
    youtube_patterns = [
        r'^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+&list=[\w-]+',
        r'^https?://(?:www\.)?youtube\.com/playlist\?list=[\w-]+',
        r'^https?://(?:www\.)?youtube\.com/channel/[\w-]+',
        r'^https?://(?:www\.)?youtube\.com/c/[\w-]+',
        r'^https?://(?:www\.)?youtube\.com/user/[\w-]+',
        r'^https?://youtu\.be/[\w-]+',
    ]
    
    return any(re.match(pattern, url) for pattern in youtube_patterns)

def extract_video_id(url: str) -> Optional[str]:
    """
    Extracts the video ID from a YouTube URL.
    Returns None if the URL is invalid or doesn't contain a video ID.
    """
    patterns = [
        r'(?:v=|/v/|youtu\.be/|/embed/)([^"&?/s]{11})',
        r'(?:v=|/v/|youtu\.be/|/embed/)([^"&?/s]{11})&list=',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def download_transcript(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Downloads the transcript of a YouTube video using youtube_transcript_api.
    Returns a tuple of (transcript_text, error_message).
    If successful, error_message will be None.
    """
    if not is_valid_youtube_url(url):
        return None, "Invalid YouTube URL format"
    
    video_id = extract_video_id(url)
    if not video_id:
        return None, "Could not extract video ID from URL"
    
    try:
        # Directly fetch transcript using the video ID
        transcript_parts = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine all parts into a single text, preserving line breaks
        full_transcript = '\n'.join(part.get('text', '') for part in transcript_parts)
        
        return full_transcript, None
            
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video"
    except NoTranscriptFound:
        return None, "No transcript available for this video"
    except Exception as e:
        logger.error(f"Error downloading transcript: {str(e)}", exc_info=True)
        return None, f"Error downloading transcript: {str(e)}" 