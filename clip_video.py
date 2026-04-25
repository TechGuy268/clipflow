import subprocess
import os
import logging

log = logging.getLogger(__name__)

MIN_CLIP_DURATION = 15.0
TARGET_SHORT_CLIP = 30.0
MIN_VALID_OUTPUT_BYTES = 1024


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


def _seconds_to_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


def _run_ffmpeg(args: list):
    subprocess.run(args, check=True, capture_output=True)


def _ffmpeg_copy(input_path: str, start_s: float, duration_s: float, output_path: str):
    _run_ffmpeg([
        "ffmpeg",
        "-ss", f"{start_s:.3f}",
        "-i", input_path,
        "-t", f"{duration_s:.3f}",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts",
        "-movflags", "+faststart",
        output_path,
        "-y",
    ])


def _ffmpeg_reencode(input_path: str, start_s: float, duration_s: float, output_path: str):
    _run_ffmpeg([
        "ffmpeg",
        "-ss", f"{start_s:.3f}",
        "-i", input_path,
        "-t", f"{duration_s:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
        "-y",
    ])


def extract_clip(input_path: str, start_s: float, duration_s: float, output_path: str) -> str:
    """Try fast stream-copy; fall back to re-encode if copy fails or yields a bad file."""
    try:
        _ffmpeg_copy(input_path, start_s, duration_s, output_path)
        if os.path.exists(output_path) and os.path.getsize(output_path) >= MIN_VALID_OUTPUT_BYTES:
            return output_path
        log.warning(f"EXTRACT copy produced empty/tiny file at {output_path}, falling back to re-encode")
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")[-300:]
        log.warning(f"EXTRACT copy failed ({stderr}), falling back to re-encode")

    _ffmpeg_reencode(input_path, start_s, duration_s, output_path)
    if not os.path.exists(output_path) or os.path.getsize(output_path) < MIN_VALID_OUTPUT_BYTES:
        raise RuntimeError(f"ffmpeg produced no usable output at {output_path}")
    return output_path


def extract_all_clips(input_path: str, highlights: list, output_dir: str, name_prefix: str = "clip") -> list:
    """Extract clips and return [{'index': i, 'path': '...'}, ...] for each highlight that produced a file."""
    os.makedirs(output_dir, exist_ok=True)
    video_duration = get_video_duration(input_path)
    log.info(f"EXTRACT video_duration={video_duration:.2f}s, highlights={len(highlights)}")
    results = []

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

        output_path = os.path.join(output_dir, f"{name_prefix}_{i+1}.mp4")
        log.info(f"EXTRACT clip {i+1} {start_s:.2f}..{end_s:.2f} ({duration_s:.2f}s) -> {output_path}")
        try:
            extract_clip(input_path, start_s, duration_s, output_path)
            results.append({"index": i, "path": output_path})
        except (subprocess.CalledProcessError, RuntimeError) as e:
            stderr = ""
            if isinstance(e, subprocess.CalledProcessError):
                stderr = (e.stderr or b"").decode("utf-8", errors="replace")[-300:]
            log.error(f"EXTRACT clip {i+1} failed: {e} {stderr}")

    log.info(f"EXTRACT done, produced {len(results)} clips")
    return results
