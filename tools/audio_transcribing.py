# audio_transcribing.py
from langchain.tools import tool
import speech_recognition as sr
from pydub import AudioSegment
import os
import requests
import logging
import tempfile
from urllib.parse import urlparse
import mimetypes

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LLM_DIR = "LLMFiles"
os.makedirs(LLM_DIR, exist_ok=True)

def _download_to_temp(url: str, timeout: int = 30):
    """Download a URL to a temp file and return its path and suggested extension."""
    try:
        r = requests.get(url, timeout=timeout)
    except requests.RequestException as e:
        logger.warning("Download failed for %s: %s", url, e)
        return None, f"download_error:{e}"

    if r.status_code != 200:
        logger.info("Download returned HTTP %s for %s", r.status_code, url)
        return None, f"http_{r.status_code}"

    # determine extension from URL or content-type
    parsed = urlparse(url)
    ext = None
    if parsed.path and "." in parsed.path:
        ext = os.path.splitext(parsed.path)[1]
    if not ext:
        content_type = r.headers.get("content-type", "")
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip() or "") or ".bin"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(r.content)
    tmp.flush()
    tmp.close()
    return tmp.name, None

def _ensure_wav(input_path: str):
    """
    Ensure the audio is in WAV format. Return path to wav file (may be same if already wav)
    or (None, error_str).
    """
    try:
        # pydub will call ffmpeg/ffprobe under the hood; catch errors if missing
        if input_path.lower().endswith(".wav"):
            return input_path, None

        # try to load generically (pydub auto-detects format)
        audio = AudioSegment.from_file(input_path)
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav.close()
        audio.export(tmp_wav.name, format="wav")
        return tmp_wav.name, None
    except Exception as e:
        logger.exception("Failed to convert audio to wav (ffmpeg missing or corrupt file): %s", e)
        return None, f"pydub_conversion_error:{e}"

@tool
def transcribe_audio(source: str) -> str:
    """
    Transcribe an audio file (local filename under LLMFiles or an HTTP/HTTPS URL).

    Args:
        source: either a local filename (relative to LLMFiles/) or a URL starting with http(s).

    Returns:
        str: the transcribed text, or an error string starting with 'AUDIO_ERROR:' on failure.
    """
    # Resolve local path or download
    is_url = source.lower().startswith("http://") or source.lower().startswith("https://")
    temp_files = []

    try:
        if is_url:
            logger.info("transcribe_audio: downloading URL: %s", source)
            downloaded, err = _download_to_temp(source)
            if err:
                return f"AUDIO_ERROR:{err}"
            audio_path = downloaded
            temp_files.append(audio_path)
        else:
            # treat as local path under LLMFiles
            local_path = os.path.join(LLM_DIR, source)
            if not os.path.exists(local_path):
                logger.info("transcribe_audio: local file not found: %s", local_path)
                return f"AUDIO_ERROR:local_not_found"
            audio_path = local_path

        # convert to wav if necessary
        wav_path, conv_err = _ensure_wav(audio_path)
        if conv_err:
            return f"AUDIO_ERROR:{conv_err}"
        if wav_path != audio_path:
            temp_files.append(wav_path)

        # perform speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as src:
            audio_data = recognizer.record(src)

        try:
            text = recognizer.recognize_google(audio_data)
            text = (text or "").strip()
            if text == "":
                return "AUDIO_ERROR:empty_transcript"
            return text
        except sr.UnknownValueError:
            logger.info("recognize_google could not understand audio")
            return "AUDIO_ERROR:unknown_value"
        except sr.RequestError as e:
            logger.warning("recognize_google request error: %s", e)
            return f"AUDIO_ERROR:recognizer_request_error:{e}"

    except Exception as e:
        logger.exception("Unexpected error in transcribe_audio: %s", e)
        return f"AUDIO_ERROR:unexpected:{e}"
    finally:
        # Cleanup only the temporary files we created
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
