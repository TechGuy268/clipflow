import os
import json
import re
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a world-class video editor specialising in short-form content for YouTube Shorts, TikTok, Instagram Reels, and LinkedIn.

Your job is to identify the best standalone moments from any type of video — interviews, podcasts, speeches, tutorials, music performances, vlogs, or anything else. You always return exactly the number of clips requested, picking the strongest moments available regardless of content type."""


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


def parse_highlights(raw: str) -> list:
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().strip("`").strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return json.loads(raw)


def detect_highlights(segments: list, num_clips: int = 5) -> list:
    transcript = format_transcript(segments)

    prompt = f"""Analyse this video transcript and identify the {num_clips} best moments for standalone short-form clips.

This could be any type of video — spoken content, music, performance, tutorial, vlog, etc. Adapt your selection accordingly.

SELECTION CRITERIA:
1. Best standalone moments — sections that work on their own without needing context
2. High energy or emotional impact — engaging, surprising, funny, moving, or punchy
3. Natural boundaries — starts and ends at clean points, not mid-sentence or mid-phrase
4. Good length — target 30–90 seconds per clip

IMPORTANT: Always return exactly {num_clips} clips. Pick the best {num_clips} moments available, even if the content is music or non-spoken video. Never return an empty array.

Rate each clip 1–10. Return ALL {num_clips} clips sorted best first.

OUTPUT: Return ONLY a valid JSON array, no explanation, no markdown:
[{{"start": "HH:MM:SS", "end": "HH:MM:SS", "title": "Short clip title", "reason": "One sentence on why this moment works", "score": 8}}]

TRANSCRIPT:
{transcript}"""

    for attempt, temp in enumerate([0.3, 0.2, 0.1]):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
            )
            highlights = parse_highlights(response.choices[0].message.content)
            highlights.sort(key=lambda x: x.get("score", 0), reverse=True)
            return highlights[:num_clips]
        except (json.JSONDecodeError, ValueError, AttributeError):
            if attempt == 2:
                raise RuntimeError("AI returned an invalid response after 3 attempts. Please try again.")
