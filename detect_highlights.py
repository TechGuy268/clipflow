import os
import json
import re
import logging
from groq import Groq

log = logging.getLogger(__name__)
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


def evenly_spaced_fallback(segments: list, num_clips: int) -> list:
    """Generate evenly-spaced clips across the video when the LLM returns nothing usable."""
    if not segments:
        return []
    total = max(seg["end"] for seg in segments)
    target_len = 45.0
    if num_clips * target_len > total:
        target_len = max(20.0, total / max(num_clips, 1))
    clips = []
    for i in range(num_clips):
        center = total * (i + 1) / (num_clips + 1)
        start = max(0.0, center - target_len / 2)
        end = min(total, start + target_len)
        clips.append({
            "start": seconds_to_timestamp(start),
            "end": seconds_to_timestamp(end),
            "title": f"Highlight {i + 1}",
            "reason": "Evenly-spaced fallback selection.",
            "score": 6,
        })
    return clips


def detect_highlights(segments: list, num_clips: int = 5) -> list:
    transcript = format_transcript(segments)
    log.info(f"DETECT segments={len(segments)} num_clips={num_clips}")

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
            raw = response.choices[0].message.content
            log.info(f"DETECT attempt {attempt+1} raw_len={len(raw or '')}")
            highlights = parse_highlights(raw)
            log.info(f"DETECT attempt {attempt+1} parsed {len(highlights)} highlights")
            if not highlights:
                log.warning(f"DETECT attempt {attempt+1} empty list, retrying" if attempt < 2 else "DETECT empty after retries, using fallback")
                if attempt < 2:
                    continue
                return evenly_spaced_fallback(segments, num_clips)
            highlights.sort(key=lambda x: x.get("score", 0), reverse=True)
            return highlights[:num_clips]
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            log.warning(f"DETECT attempt {attempt+1} parse failed: {e}")
            if attempt == 2:
                log.warning("DETECT all parse attempts failed, using fallback")
                return evenly_spaced_fallback(segments, num_clips)
