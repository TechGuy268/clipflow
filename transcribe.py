import whisper


def transcribe_video(video_path: str, model_size: str = "base") -> list:
    model = whisper.load_model(model_size)
    result = model.transcribe(video_path)
    return result["segments"]
