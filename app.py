import os
import uuid
import threading
import zipfile
import io
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
from transcribe import transcribe_video
from detect_highlights import detect_highlights
from clip_video import extract_all_clips

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}

jobs = {}


def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def process_job(job_id: str, video_path: str, num_clips: int):
    try:
        jobs[job_id].update({"stage": "transcribing", "progress": 15})
        segments = transcribe_video(video_path)

        jobs[job_id].update({"stage": "analyzing", "progress": 55})
        highlights = detect_highlights(segments, num_clips=num_clips)

        jobs[job_id].update({"stage": "extracting", "progress": 75})
        clip_paths = extract_all_clips(video_path, highlights, OUTPUT_FOLDER)

        jobs[job_id].update({
            "stage": "done",
            "progress": 100,
            "clips": [os.path.basename(p) for p in clip_paths],
            "highlights": highlights,
        })
    except Exception as e:
        jobs[job_id].update({"stage": "error", "error": str(e)})
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify({"error": "No video file provided"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use .mp4, .mov, .mkv, or .webm"}), 400

    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    file.save(video_path)

    num_clips = max(1, min(20, int(request.form.get("num_clips", 5))))
    jobs[job_id] = {"stage": "queued", "progress": 5}

    thread = threading.Thread(target=process_job, args=(job_id, video_path, num_clips), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


@app.route("/download-all/<job_id>")
def download_all(job_id):
    job = jobs.get(job_id)
    if not job or job.get("stage") != "done":
        return jsonify({"error": "Clips not ready"}), 400

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for clip in job.get("clips", []):
            path = os.path.join(OUTPUT_FOLDER, clip)
            if os.path.exists(path):
                zf.write(path, clip)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name="clipflow_highlights.zip",
    )


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
