import os

PIPER_MODELS_LOCATION = os.environ.get("PIPER_MODELS_LOCATION", "/data/piper-models")
PIPER_TTS_MODEL = os.environ.get("PIPER_TTS_MODEL", "en_US-amy-medium")

SPEECH_PIPELINE_SAMPLERATE = 16000

MUMBLE_SERVER_HOST = os.environ.get("MUMBLE_SERVER_HOST", "mumble")
MUMBLE_SERVER_PORT = int(os.environ.get("MUMBLE_SERVER_PORT", 64738))
MUMBLE_SERVER_PASSWORD = os.environ.get("MUMBLE_SERVER_PASSWORD", "")

WHISPER_MODELS_LOCATION = os.environ.get(
    "WHISPER_MODELS_LOCATION", "/data/whisper-models"
)
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "small.en")
WHISPER_USE_DEVICE = os.environ.get("WHISPER_USE_DEVICE", "cpu")
WHISPER_USE_COMPUTE_TYPE = os.environ.get("WHISPER_USE_COMPUTE_TYPE", "int8")

ASSISTANT_NAME = os.environ.get("ASSISTANT_NAME", "Aya")
ASSISTANT_BREAK_ON_TOKENS = [".", ",", "!", "?", ";", ":", "\n"]

VAD_SILENCE_THRESHOLD = int(os.environ.get("VAD_SILENCE_THRESHOLD", 16))
VAD_LOG_AUDIO_DIRECTORY = os.environ.get("VAD_LOG_AUDIO_DIRECTORY", "/data/vad-records")

OLLAMA_LLM = os.environ.get("OLLAMA_LLM", "mistral:7b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_LLM_STOP_TOKENS = [f"{ASSISTANT_NAME}:", f"{ASSISTANT_NAME}:".lower()]
OLLAMA_LLM_TEMPERATURE = int(os.environ.get("OLLAMA_LLM_TEMPERATURE", 0))
