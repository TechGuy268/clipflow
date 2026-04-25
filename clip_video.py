import subprocess
import os
import logging

log = logging.getLogger(__name__)

MIN_CLIP_DURATION = 15.0
TARGET_SHORT_CLIP = 30.0


def timestamp_to_seconds(ts: str) -> float:
    parts = ts.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def get_video_duration(input_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def expand_window(start_s: float, end_s: float, video_duration: float, target: float = TARGET_SHORT_CLIP):
    """Expand a too-short window around its midpoint, clamped to video bounds."""
    duration = end_s - start_s
    if duration >= MIN_CLIP_DURATION:
        return start_s, end_s
    midpoint = (start_s + end_s) / 2
    half = target / 2
    new_start = max(0.0, midpoint - half)
    new_end = min(video_duration, midpoint + half)
    if new_end - new_start < MIN_CLIP_DURATION:
        new_start = max(0.0, new_end - MIN_CLIP_DURATION)
        new_end = min(video_duration, new_start + MIN_CLIP_DURATION)
    return new_start, new_end


def extract_clip(input_path: str, start_s: float, duration_s: float, output_path: str) -> str:
    command = [
        "ffmpeg",
        "-ss", f"{start_s:.3f}",
        "-i", input_path,
        "-t", f"{duration_s:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-movflags", "+faststart",
        output_path,
        "-y",
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_path


def extract_all_clips(input_path: str, highlights: list, output_dir: str) -> list:
    os.makedirs(output_dir, exist_ok=True)
    video_duration = get_video_duration(input_path)
    log.info(f"EXTRACT video_duration={video_duration:.2f}s, highlights={len(highlights)}")
    clip_paths = []

    for i, clip in enumerate(highlights):
        try:
            raw_start = max(0.0, timestamp_to_seconds(clip["start"]))
            raw_end = min(video_duration, timestamp_to_seconds(clip["end"]))
        except (KeyError, ValueError, AttributeError) as e:
            log.warning(f"EXTRACT clip {i+1} bad timestamps {clip.get('start')}..{clip.get('end')}: {e}")
            continue

        if raw_end <= raw_start:
            log.warning(f"EXTRACT clip {i+1} inverted/zero range {raw_start:.2f}..{raw_end:.2f}, skipping")
            continue

        start_s, end_s = expand_window(raw_start, raw_end, video_duration)
        duration_s = end_s - start_s

        clip["start"] = _seconds_to_ts(start_s)
        clip["end"] = _seconds_to_ts(end_s)

        output_path = os.path.join(output_dir, f"clip_{i+1}.mp4")
        log.info(f"EXTRACT clip {i+1} {start_s:.2f}..{end_s:.2f} ({duration_s:.2f}s) -> {output_path}")
        try:
            extract_clip(input_path, start_s, duration_s, output_path)
            clip_paths.append(output_path)
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or b"").decode("utf-8", errors="replace")[-500:]
            log.error(f"EXTRACT clip {i+1} ffmpeg failed: {stderr}")

    log.info(f"EXTRACT done, produced {len(clip_paths)} clips")
    return clip_paths


def _seconds_to_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"
