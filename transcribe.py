import os
import subprocess
import tempfile
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def extract_audio(video_path: str, audio_path: str):
    """Extract low-bitrate mono audio from video for transcription."""
    subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vn",
            "-acodec", "libmp3lame",
            "-ar", "16000",
            "-ac", "1",
            "-b:a", "64k",
            audio_path,
            "-y",
        ],
        check=True,
        capture_output=True,
    )


def transcribe_video(video_path: str, **kwargs) -> list:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        audio_path = tmp.name

    try:
        extract_audio(video_path, audio_path)

        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        return [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in response.segments
        ]
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
