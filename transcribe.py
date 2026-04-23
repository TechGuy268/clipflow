import os
import whisper

_model_cache = {}


def transcribe_video(video_path: str, model_size: str = None) -> list:
    model_size = model_size or os.environ.get("WHISPER_MODEL", "small")
    if model_size not in _model_cache:
        _model_cache[model_size] = whisper.load_model(model_size)
    model = _model_cache[model_size]
    result = model.transcribe(
        video_path,
        verbose=False,
        condition_on_previous_text=True,
        temperature=0.0,
    )
    return result["segments"]
