import subprocess
import os


def extract_clip(input_path: str, start: str, end: str, output_path: str) -> str:
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ss", start,
        "-to", end,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_path,
        "-y"
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_path


def extract_all_clips(input_path: str, highlights: list, output_dir: str) -> list:
    os.makedirs(output_dir, exist_ok=True)
    clip_paths = []

    for i, clip in enumerate(highlights):
        output_path = os.path.join(output_dir, f"clip_{i+1}.mp4")
        extract_clip(input_path, clip["start"], clip["end"], output_path)
        clip_paths.append(output_path)

    return clip_paths
