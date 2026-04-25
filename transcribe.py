import os
import time
import logging
import subprocess
import tempfile
from groq import Groq

log = logging.getLogger(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

GROQ_MAX_BYTES = 24 * 1024 * 1024


def extract_audio(video_path: str, audio_path: str):
    """Extract low-bitrate mono audio from video for transcription."""
    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "48k",
            audio_path,
            "-y",
        ],
        check=True,
        capture_output=True,
    )


def transcribe_video(video_path: str) -> list:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        audio_path = tmp.name

    try:
        extract_audio(video_path, audio_path)
        size = os.path.getsize(audio_path)
        log.info(f"TRANSCRIBE audio_size={size} bytes")

        if size > GROQ_MAX_BYTES:
            raise RuntimeError(
                f"Audio is too long for transcription ({size // (1024*1024)}MB extracted, "
                f"limit is {GROQ_MAX_BYTES // (1024*1024)}MB). Try a shorter video."
            )

        last_err = None
        for attempt in range(2):
            try:
                with open(audio_path, "rb") as f:
                    response = client.audio.transcriptions.create(
                        file=(os.path.basename(audio_path), f),
                        model="whisper-large-v3-turbo",
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                        temperature=0,
                    )
                segments = [
                    {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
                    for seg in response.segments
                ]
                log.info(f"TRANSCRIBE produced {len(segments)} segments")
                return segments
            except Exception as e:
                last_err = e
                log.warning(f"TRANSCRIBE attempt {attempt+1} failed: {e}")
                if attempt == 0:
                    time.sleep(2)
        raise last_err
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
