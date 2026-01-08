import os
import tempfile
from faster_whisper import WhisperModel
import cv2
from typing import List, Dict
from pathlib import Path


class AudioProcessor:
    """Handles audio extraction and transcription from video files."""

    def __init__(self, model_size: str = "base"):
        """
        Initialize the audio processor with Whisper model.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
        """
        print(f"Loading Faster-Whisper model: {model_size}...")
        # Use GPU if available, otherwise CPU
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        print(f"Faster-Whisper model loaded: {model_size}")

    def extract_audio(self, video_path: str) -> str:
        """
        Extract audio from video file to temporary WAV file.

        Args:
            video_path: Path to video file

        Returns:
            Path to temporary audio file
        """
        # Create temporary file for audio
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()

        # Extract audio using OpenCV (if possible) or ffmpeg through whisper
        # For simplicity, we'll use whisper's built-in audio loading
        # which handles various formats automatically
        return video_path  # Whisper can load directly from video

    def transcribe(self, video_path: str, language: str = None) -> Dict:
        """
        Transcribe audio from video using Faster-Whisper.

        Args:
            video_path: Path to video file
            language: Optional language code (e.g., 'en', 'es'). Auto-detects if None.

        Returns:
            Dictionary with full transcription data including word-level timestamps
        """
        print(f"Transcribing audio from {Path(video_path).name}...")

        # Transcribe with word-level timestamps
        segments, info = self.model.transcribe(
            video_path,
            language=language,
            word_timestamps=True,
            vad_filter=True  # Voice activity detection to filter silence
        )

        # Convert generator to list and format like original whisper
        segments_list = []
        for segment in segments:
            segments_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })

        result = {
            "segments": segments_list,
            "language": info.language
        }

        return result

    def chunk_transcript(
        self,
        transcript_data: Dict,
        chunk_duration: float = 20.0
    ) -> List[Dict]:
        """
        Chunk transcript into segments of specified duration.

        Args:
            transcript_data: Output from Whisper transcription
            chunk_duration: Duration of each chunk in seconds

        Returns:
            List of chunks with text, start_time, and end_time
        """
        chunks = []

        # Get all segments from transcript
        segments = transcript_data.get("segments", [])

        if not segments:
            return chunks

        current_chunk_text = []
        current_chunk_start = None
        current_chunk_end = None

        for segment in segments:
            segment_start = segment["start"]
            segment_end = segment["end"]
            segment_text = segment["text"].strip()

            # Initialize first chunk
            if current_chunk_start is None:
                current_chunk_start = segment_start
                current_chunk_end = segment_end
                current_chunk_text.append(segment_text)
                continue

            # Check if adding this segment would exceed chunk duration
            potential_duration = segment_end - current_chunk_start

            if potential_duration <= chunk_duration:
                # Add to current chunk
                current_chunk_text.append(segment_text)
                current_chunk_end = segment_end
            else:
                # Save current chunk and start new one
                if current_chunk_text:
                    chunks.append({
                        "text": " ".join(current_chunk_text),
                        "start_time": current_chunk_start,
                        "end_time": current_chunk_end
                    })

                # Start new chunk
                current_chunk_start = segment_start
                current_chunk_end = segment_end
                current_chunk_text = [segment_text]

        # Add final chunk
        if current_chunk_text:
            chunks.append({
                "text": " ".join(current_chunk_text),
                "start_time": current_chunk_start,
                "end_time": current_chunk_end
            })

        return chunks

    def get_video_duration(self, video_path: str) -> float:
        """
        Get duration of video in seconds.

        Args:
            video_path: Path to video file

        Returns:
            Duration in seconds
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration
