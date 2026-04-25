import os
import io
import json
import uuid
import threading
import zipfile
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
from transcribe import transcribe_video
from detect_highlights import detect_highlights
from clip_video import extract_all_clips

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
JOBS_FOLDER   = "jobs"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


def allowed_file(filename):
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def save_job(job_id, data):
    os.makedirs(JOBS_FOLDER, exist_ok=True)
    path = os.path.join(JOBS_FOLDER, f"{job_id}.json")
    app.logger.info(f"SAVE_JOB {job_id} stage={data.get('stage')} path={os.path.abspath(path)}")
    with open(path, "w") as f:
        json.dump(data, f)
    app.logger.info(f"SAVE_JOB done, exists={os.path.exists(path)}")


def load_job(job_id):
    path = os.path.join(JOBS_FOLDER, f"{job_id}.json")
    exists = os.path.exists(path)
    app.logger.info(f"LOAD_JOB {job_id} path={os.path.abspath(path)} exists={exists}")
    if not exists:
        return None
    with open(path) as f:
        return json.load(f)


def process_job(job_id, video_path, num_clips):
    try:
        save_job(job_id, {"stage": "transcribing", "progress": 15})
        segments = transcribe_video(video_path)

        save_job(job_id, {"stage": "analyzing", "progress": 55})
        highlights = detect_highlights(segments, num_clips=num_clips)

        save_job(job_id, {"stage": "extracting", "progress": 75})
        clip_paths = extract_all_clips(video_path, highlights, OUTPUT_FOLDER)

        save_job(job_id, {
            "stage": "done",
            "progress": 100,
            "clips": [os.path.basename(p) for p in clip_paths],
            "highlights": highlights,
        })
    except Exception as e:
        save_job(job_id, {"stage": "error", "error": str(e)})
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

    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, JOBS_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
    app.logger.info(f"PROCESS saving upload to {video_path}")
    file.save(video_path)
    app.logger.info(f"PROCESS upload saved, size={os.path.getsize(video_path)}")

    num_clips = max(1, min(20, int(request.form.get("num_clips", 5))))
    save_job(job_id, {"stage": "queued", "progress": 5})
    app.logger.info(f"PROCESS job created, returning job_id={job_id}")

    thread = threading.Thread(target=process_job, args=(job_id, video_path, num_clips), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = load_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/ping")
def ping():
    return "ok"


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


@app.route("/download-all", methods=["POST"])
def download_all():
    clips = request.json.get("clips", [])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for clip in clips:
            path = os.path.join(OUTPUT_FOLDER, clip)
            if os.path.exists(path):
                zf.write(path, clip)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="clipflow_highlights.zip")


if __name__ == "__main__":
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, JOBS_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
