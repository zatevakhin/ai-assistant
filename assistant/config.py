import os

PIPER_MODELS_LOCATION = os.environ.get("PIPER_MODELS_LOCATION", "data/piper-models")
PIPER_TTS_MODEL = os.environ.get("PIPER_TTS_MODEL", "en_US-amy-medium")

SPEECH_PIPELINE_SAMPLERATE = 16000
SPEECH_PIPELINE_BUFFER_SIZE_MILIS = 32

MUMBLE_SERVER_HOST = os.environ.get("MUMBLE_SERVER_HOST", "127.0.0.1")
MUMBLE_SERVER_PORT = int(os.environ.get("MUMBLE_SERVER_PORT", 64738))
MUMBLE_SERVER_PASSWORD = os.environ.get("MUMBLE_SERVER_PASSWORD", "")

WHISPER_MODELS_LOCATION = os.environ.get(
    "WHISPER_MODELS_LOCATION", "data/whisper-models"
)
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "small.en")
WHISPER_USE_DEVICE = os.environ.get("WHISPER_USE_DEVICE", "cpu")
WHISPER_USE_COMPUTE_TYPE = os.environ.get("WHISPER_USE_COMPUTE_TYPE", "int8")

ASSISTANT_NAME = os.environ.get("ASSISTANT_NAME", "Aya")
ASSISTANT_BREAK_ON_TOKENS = [".", ",", "!", "?", ";", ":", "\n"]

VAD_SILENCE_THRESHOLD = int(os.environ.get("VAD_SILENCE_THRESHOLD", 16))
VAD_LOG_AUDIO_DIRECTORY = os.environ.get("VAD_LOG_AUDIO_DIRECTORY", "data/vad-records")

OLLAMA_LLM = os.environ.get("OLLAMA_LLM", "llama3:8b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_LLM_STOP_TOKENS = [f"{ASSISTANT_NAME}:", f"{ASSISTANT_NAME}:".lower()]
OLLAMA_LLM_TEMPERATURE = int(os.environ.get("OLLAMA_LLM_TEMPERATURE", 0.2))

TOPIC_MUMBLE_SOUND_NEW = "mumble/got/audio"
TOPIC_MUMBLE_PLAY_AUDIO = "mumble/play/audio"
TOPIC_MUMBLE_INTERRUPT_AUDIO = "mumble/interrupt/audio"

TOPIC_SPEECH_SYNTHESIS_INTERRUPT = "speech/synthesis/interrupt"

TOPIC_VAD_SPEECH_NEW = "vad/speech/new"
TOPIC_TRANSCRIPTION_DONE = "transcription/done"
TOPIC_LLM_TOKEN_NEW = "llm/token/new"
TOPIC_LLM_STREAM_DONE = "llm/stream/done"
TOPIC_LLM_ON_SENTENCE = "llm/on/sentence"
TOPIC_LLM_STREAM_INTERRUPT = "llm/stream/interrupt"

ZENOH_CONFIG = {
    "connect": {
        "endpoints": ["tcp/localhost:7447"],
    },
    "listen": {
        "endpoints": ["tcp/localhost:7447"],
    },
}
