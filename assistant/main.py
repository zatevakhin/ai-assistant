#!/usr/bin/python3
import time
from functools import partial

import numpy as np
import pymumble_py3
import resampy
from faster_whisper import WhisperModel
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from pymumble_py3.callbacks import PYMUMBLE_CLBK_SOUNDRECEIVED
from pymumble_py3.constants import PYMUMBLE_SAMPLERATE
from pymumble_py3.soundqueue import SoundChunk
from voice_forge import PiperTts
from voice_pulse.config import Config
from voice_pulse.enums import VadEngine
from voice_pulse.input_sources import CallbackInput
from voice_pulse.listener import Listener

from assistant.audio import AudioBufferHandler, record_audio_chunk
from assistant.config import (
    ASSISTANT_BREAK_ON_TOKENS,
    ASSISTANT_NAME,
    MUMBLE_SERVER_HOST,
    MUMBLE_SERVER_PASSWORD,
    MUMBLE_SERVER_PORT,
    OLLAMA_BASE_URL,
    OLLAMA_LLM,
    OLLAMA_LLM_STOP_TOKENS,
    OLLAMA_LLM_TEMPERATURE,
    PIPER_MODELS_LOCATION,
    PIPER_TTS_MODEL,
    SPEECH_PIPELINE_SAMPLERATE,
    VAD_LOG_AUDIO_DIRECTORY,
    VAD_SILENCE_THRESHOLD,
    WHISPER_MODEL_NAME,
    WHISPER_MODELS_LOCATION,
    WHISPER_USE_COMPUTE_TYPE,
    WHISPER_USE_DEVICE,
)

MY_SAMPLERATE = 16000


def sound_received_handler(
    handler: AudioBufferHandler, user: str, soundchunk: SoundChunk
):
    if user in [ASSISTANT_NAME]:
        return

    handler(soundchunk)


if __name__ == "__main__":
    whisper = WhisperModel(
        WHISPER_MODEL_NAME,
        device=WHISPER_USE_DEVICE,
        compute_type=WHISPER_USE_COMPUTE_TYPE,
        download_root=WHISPER_MODELS_LOCATION,
    )
    tts = PiperTts(PIPER_TTS_MODEL, PIPER_MODELS_LOCATION)

    output_parser = StrOutputParser()
    llm = Ollama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_LLM,
        temperature=OLLAMA_LLM_TEMPERATURE,
    )
    prompt = PromptTemplate.from_template(
        f"You are a helpful AI assistant. Your name is {ASSISTANT_NAME}. Your answers always short and concise.\nuser: {{query}}\n{ASSISTANT_NAME}: "
    )
    chain = prompt | llm.bind(stop=OLLAMA_LLM_STOP_TOKENS) | output_parser

    config = Config(
        vad_engine=VadEngine.SILERIO,
        block_duration=32,
        silence_threshold=VAD_SILENCE_THRESHOLD,
    )
    stream = CallbackInput(config.blocksize)

    mumble = pymumble_py3.Mumble(
        host=MUMBLE_SERVER_HOST,
        user=ASSISTANT_NAME,
        port=MUMBLE_SERVER_PORT,
        password=MUMBLE_SERVER_PASSWORD,
    )

    audio_handler = AudioBufferHandler(
        stream.receive_chunk, SPEECH_PIPELINE_SAMPLERATE, 32
    )

    mumble.callbacks.set_callback(
        PYMUMBLE_CLBK_SOUNDRECEIVED, partial(sound_received_handler, audio_handler)
    )
    mumble.set_receive_sound(True)
    mumble.start()
    time.sleep(1)

    for speech in Listener(config, stream):
        # record_audio_chunk(
        #     record_audio_chunk,
        #     directory=VAD_LOG_AUDIO_DIRECTORY,
        #     samplerate=SPEECH_PIPELINE_SAMPLERATE,
        # )

        print(f"speech ({type(speech)})", len(speech))

        segments, info = whisper.transcribe(speech)

        print(f"> {info.language} ({info.language_probability})")

        text = ". ".join(
            map(lambda seg: seg.text, filter(lambda seg: seg.text, segments))
        ).strip()
        print(">", text)

        if info.language in ["en"] and info.language_probability > 0.6:
            bufferized_tokens = []
            for token in chain.stream({"query": text}):
                bufferized_tokens.append(token)

                if token in ASSISTANT_BREAK_ON_TOKENS:
                    data, sr = tts.synthesize_stream("".join(bufferized_tokens))
                    resampled_data = resampy.resample(data, sr, PYMUMBLE_SAMPLERATE)
                    mumble.sound_output.add_sound(
                        resampled_data.astype(np.int16).tobytes()
                    )
                    bufferized_tokens = []

            data, sr = tts.synthesize_stream("".join(bufferized_tokens))
            resampled_data = resampy.resample(data, sr, PYMUMBLE_SAMPLERATE)
            mumble.sound_output.add_sound(resampled_data.astype(np.int16).tobytes())
            bufferized_tokens = []
