import json
import queue
import os
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import socket
import time
from dataclasses import dataclass, asdict
from vosk import Model, KaldiRecognizer

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000
VOSK_MODEL_PATH = "models/vosk-model-en-us-0.22"

# ==============================
# RESULT STRUCTURE
# ==============================
@dataclass
class STTResult:
    text: str
    confidence: float  # -1.0 means engine does not expose confidence
    loudness: float
    engine: str
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)


# ==============================
# INTERNET CHECK
# ==============================
def has_internet(host="8.8.8.8", port=53, timeout=2):
    try:
        # FIX: use create_connection as context manager — no socket leak,
        # and narrow except to OSError only (was bare except before)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ==============================
# GOOGLE STT
# ==============================
class GoogleSTT:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def listen(self, on_result=None):
        with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
            print("🎙️  Listening (Google)...")
            while True:
                audio = self.recognizer.listen(source)
                try:
                    text = self.recognizer.recognize_google(audio)
                    raw_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
                    loudness = float(
                        np.sqrt(np.mean(raw_data.astype(np.float32) ** 2))
                    )
                    result = STTResult(
                        text=text.lower(),
                        # FIX: was 1.0 (misleading "perfect confidence")
                        # -1.0 is the sentinel meaning "not available"
                        confidence=-1.0,
                        loudness=round(loudness, 2),
                        engine="google",
                        timestamp=time.time(),
                    )
                    if on_result:
                        on_result(result)
                    else:
                        print(result.to_dict())

                # FIX: was bare except — now handles each case explicitly
                except sr.UnknownValueError:
                    pass  # silence/noise — nothing to transcribe, normal
                except sr.RequestError as e:
                    print(f"⚠️  Google STT API error: {e}")
                    time.sleep(2)  # back-off before retrying


# ==============================
# VOSK STT
# ==============================
class VoskSTT:
    def __init__(self):
        # FIX: validate model path before attempting to load — previously
        # a bad path produced a cryptic C++ crash with no useful message
        if not os.path.exists(VOSK_MODEL_PATH):
            raise FileNotFoundError(
                f"Vosk model not found at {VOSK_MODEL_PATH!r}. "
                "Download a model from https://alphacephei.com/vosk/models "
                "and extract it to that path."
            )
        print("🧠 Loading Vosk model...")
        self.model = Model(VOSK_MODEL_PATH)
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.audio_queue = queue.Queue()

    # FIX: renamed time_info → time_obj to avoid shadowing the `time` module
    def _callback(self, indata, frames, time_obj, status):
        self.audio_queue.put(bytes(indata))

    def _extract_confidence(self, result_json):
        words = result_json.get("result", [])
        if not words:
            return 0.0
        conf_values = [w.get("conf", 0.0) for w in words]
        return round(sum(conf_values) / len(conf_values), 2)

    def listen(self, on_result=None):
        print("🎙️  Listening (Vosk offline)...")
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=1,
            callback=self._callback,
        ):
            # FIX: accumulate audio chunks so loudness reflects the full
            # utterance, not just the last queued block (was misleading before)
            audio_buffer = []

            while True:
                data = self.audio_queue.get()
                raw_audio = np.frombuffer(data, dtype=np.int16)
                audio_buffer.append(raw_audio)

                if self.recognizer.AcceptWaveform(data):
                    result_json = json.loads(self.recognizer.Result())
                    text = result_json.get("text", "")

                    if text.strip():
                        # Compute RMS over the entire utterance
                        full_audio = np.concatenate(audio_buffer)
                        loudness = float(
                            np.sqrt(np.mean(full_audio.astype(np.float32) ** 2))
                        ) if len(full_audio) > 0 else 0.0

                        confidence = self._extract_confidence(result_json)
                        result = STTResult(
                            text=text.lower(),
                            confidence=confidence,
                            loudness=round(loudness, 2),
                            engine="vosk",
                            timestamp=time.time(),
                        )
                        if on_result:
                            on_result(result)
                        else:
                            print(result.to_dict())

                    # Reset buffer after each recognised utterance
                    audio_buffer.clear()


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    engine = None
    try:
        if has_internet():
            print("🌐 Internet detected → Using Google")
            engine = GoogleSTT()
        else:
            print("🚫 No internet → Using Vosk")
            engine = VoskSTT()

        # FIX: Ctrl+C is now handled — previously threw a raw traceback
        engine.listen()

    except FileNotFoundError as e:
        print(f"❌ {e}")
    except KeyboardInterrupt:
        print("\n👋 Stopped.")