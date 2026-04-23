import os
import json
import re
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a world-class video editor and content strategist specialising in viral short-form content. You deeply understand what makes moments shareable on YouTube Shorts, TikTok, Instagram Reels, and LinkedIn.

Your sole job is to identify the highest-value standalone moments in a video transcript. You think like a top creator who knows exactly what hooks viewers in the first 3 seconds and delivers genuine value within 60–90 seconds. You are ruthlessly selective — you only surface moments that are truly exceptional."""


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
    # Extract JSON array if surrounded by other text
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return json.loads(raw)


def detect_highlights(segments: list, num_clips: int = 5) -> list:
    transcript = format_transcript(segments)

    prompt = f"""Analyse this video transcript and identify the {num_clips} best moments for standalone short-form clips.

SELECTION CRITERIA (in order of importance):
1. Strong hook — the clip must open with a bold statement, a surprising fact, a question, or mid-action. Never starts with filler ("So...", "Um...", "As I was saying...") or references prior context.
2. Fully self-contained — makes complete sense without watching anything else. Zero references to "as I mentioned", "earlier I said", "like we discussed".
3. High value density — packs genuine insight, a surprising reveal, actionable advice, or real emotion into a short window.
4. Clean natural ending — ends on a conclusion, punchline, or definitive pause. Never cuts mid-sentence or mid-thought.
5. Emotional pull — the moment is funny, inspiring, shocking, deeply relatable, or genuinely useful on its own.

CLIP LENGTH: Target 45–75 seconds. Hard limits: 30s minimum, 90s maximum. If a natural moment runs slightly over, prefer a clean cut over an arbitrary one.

SCORING: Rate each clip 1–10 on standalone virality potential. Be honest — most moments score 5–6. Only surface clips scoring 7 or above. If fewer than {num_clips} moments qualify, return only the ones that do.

OUTPUT: Return ONLY a valid JSON array. No explanation, no markdown, no code fences.
Format: [{{"start": "HH:MM:SS", "end": "HH:MM:SS", "title": "Short punchy clip title", "reason": "One sharp sentence on why this clip works", "score": 8}}]

TRANSCRIPT:
{transcript}"""

    for attempt, temp in enumerate([0.2, 0.1, 0.0]):
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
            highlights = [h for h in highlights if h.get("score", 0) >= 7]
            highlights.sort(key=lambda x: x.get("score", 0), reverse=True)
            return highlights[:num_clips]
        except (json.JSONDecodeError, ValueError, AttributeError):
            if attempt == 2:
                raise RuntimeError("AI returned an invalid response after 3 attempts. Please try again.")
