import subprocess
import os


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


def extract_clip(input_path: str, start_s: float, duration_s: float, output_path: str) -> str:
    command = [
        "ffmpeg",
        "-ss", f"{start_s:.3f}",
        "-i", input_path,
        "-t", f"{duration_s:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
        "-y",
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_path


def extract_all_clips(input_path: str, highlights: list, output_dir: str) -> list:
    os.makedirs(output_dir, exist_ok=True)
    video_duration = get_video_duration(input_path)
    clip_paths = []

    for i, clip in enumerate(highlights):
        start_s = max(0.0, timestamp_to_seconds(clip["start"]))
        end_s = min(video_duration, timestamp_to_seconds(clip["end"]))
        duration_s = end_s - start_s

        if duration_s < 5:
            continue

        output_path = os.path.join(output_dir, f"clip_{i+1}.mp4")
        extract_clip(input_path, start_s, duration_s, output_path)
        clip_paths.append(output_path)

    return clip_paths
