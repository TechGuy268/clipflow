import os
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, send_from_directory
from transcribe import transcribe_video
from detect_highlights import detect_highlights
from clip_video import extract_all_clips

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video file provided"}), 400

    num_clips = int(request.form.get("num_clips", 5))

    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    segments = transcribe_video(video_path, model_size="base")
    highlights = detect_highlights(segments, num_clips=num_clips)
    clip_paths = extract_all_clips(video_path, highlights, OUTPUT_FOLDER)

    return jsonify({
        "clips": [os.path.basename(p) for p in clip_paths],
        "highlights": highlights
    })


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
