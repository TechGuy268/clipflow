import os
import io
import json
import uuid
import logging
import zipfile
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, Response, stream_with_context
from werkzeug.utils import secure_filename
from transcribe import transcribe_video
from detect_highlights import detect_highlights
from clip_video import extract_all_clips

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


def allowed_file(filename):
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def event(payload: dict) -> str:
    return json.dumps(payload) + "\n"


@app.route("/")
def index():
    resp = app.make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("video")
    if not file or not file.filename:
        return jsonify({"error": "No video file provided"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use .mp4, .mov, .mkv, or .webm"}), 400

    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
    app.logger.info(f"PROCESS saving upload to {video_path}")
    file.save(video_path)
    app.logger.info(f"PROCESS upload saved, size={os.path.getsize(video_path)}")

    num_clips = max(1, min(20, int(request.form.get("num_clips", 5))))

    def generate():
        try:
            yield event({"stage": "transcribing"})
            segments = transcribe_video(video_path)

            yield event({"stage": "analyzing"})
            highlights = detect_highlights(segments, num_clips=num_clips)

            yield event({"stage": "extracting"})
            clip_paths = extract_all_clips(video_path, highlights, OUTPUT_FOLDER)

            if not clip_paths:
                yield event({"stage": "error", "error": "No clips could be extracted from this video."})
                return

            produced_basenames = {os.path.basename(p) for p in clip_paths}
            kept_highlights = [
                h for i, h in enumerate(highlights)
                if f"clip_{i+1}.mp4" in produced_basenames
            ]

            yield event({
                "stage": "done",
                "clips": [os.path.basename(p) for p in clip_paths],
                "highlights": kept_highlights,
            })
        except Exception as e:
            app.logger.exception("PROCESS failed")
            yield event({"stage": "error", "error": str(e)})
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

    resp = Response(stream_with_context(generate()), mimetype="application/x-ndjson")
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


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
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
