import os
import json
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def seconds_to_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


def format_transcript(segments: list) -> str:
    lines = []
    for seg in segments:
        start = seconds_to_timestamp(seg["start"])
        end = seconds_to_timestamp(seg["end"])
        lines.append(f"[{start} --> {end}] {seg['text'].strip()}")
    return "\n".join(lines)


def detect_highlights(segments: list, num_clips: int = 5) -> list:
    transcript = format_transcript(segments)

    prompt = f"""You are an expert video editor specialising in short-form content.
Analyse this transcript and identify the {num_clips} best moments for standalone short clips (30-90 seconds each).
Focus on: strong hooks, key insights, emotional moments, or punchy takeaways.

Return ONLY valid JSON, no explanation:
[{{ "start": "HH:MM:SS", "end": "HH:MM:SS", "reason": "brief reason" }}]

Transcript:
{transcript}
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    raw = response.choices[0].message.content
    return json.loads(raw)
