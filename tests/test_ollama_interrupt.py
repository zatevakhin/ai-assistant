import os
import sys
import threading
import time

import ollama
import pytest

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM = os.getenv("OLLAMA_LLM", "llama3.1:8b")


class StreamingInterruptTest:
    def __init__(self):
        self.interrupted = False
        self.token_count = 0
        self.streaming_complete = False
        self.stream_thread = None

    def stream_handler(self, prompt):
        try:
            print(f"Starting streaming generation with {OLLAMA_LLM}...")

            client = ollama.Client(host=OLLAMA_BASE_URL)
            for chunk in client.chat(
                model=OLLAMA_LLM,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            ):
                if self.interrupted:
                    print("\n[Manually interrupted the stream]")
                    break

                content = chunk["message"].get("content", "")
                if content:
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    self.token_count += 1

                time.sleep(0.01)

            if not self.interrupted:
                self.streaming_complete = True
                print("\n\nStreaming completed naturally")
            else:
                print("\n\nStreaming was interrupted before completion")

        except Exception as e:
            print(f"\n\nError during streaming: {str(e)}")

        print(f"Tokens received: {self.token_count}")

    def start_streaming(self, prompt):
        self.stream_thread = threading.Thread(
            target=self.stream_handler, args=(prompt,)
        )
        self.stream_thread.daemon = True
        self.stream_thread.start()

    def interrupt_streaming(self):
        self.interrupted = True


def test_streaming_interruption():
    prompt = "Write a detailed, multi-paragraph essay about the history of artificial intelligence, its current state, and future possibilities."
    print(f"\nTesting streaming interruption with {OLLAMA_LLM} at {OLLAMA_BASE_URL}")

    test = StreamingInterruptTest()
    test.start_streaming(prompt)

    interrupt_after = 2.0
    print(f"Will interrupt after {interrupt_after} seconds...")
    time.sleep(interrupt_after)

    tokens_before_interrupt = test.token_count
    test.interrupt_streaming()

    max_wait = 3.0
    wait_start = time.time()
    while test.stream_thread.is_alive() and time.time() - wait_start < max_wait:
        time.sleep(0.1)

    interrupted_successfully = not test.streaming_complete

    print("\n--- Test Results ---")
    print(f"Tokens generated before interruption: {tokens_before_interrupt}")
    print(f"Total tokens received: {test.token_count}")
    print(f"Interruption successful: {interrupted_successfully}")
    print(f"Generation completed naturally: {test.streaming_complete}")

    assert tokens_before_interrupt > 0, "No tokens were generated before interruption"
    assert interrupted_successfully, "Failed to interrupt the streaming process"


def test_streaming_completion():
    prompt = "What is artificial intelligence? Answer briefly."
    print(f"\nTesting streaming completion with {OLLAMA_LLM} at {OLLAMA_BASE_URL}")

    test = StreamingInterruptTest()
    test.start_streaming(prompt)

    max_wait = 10.0
    wait_start = time.time()
    while test.stream_thread.is_alive() and time.time() - wait_start < max_wait:
        time.sleep(0.1)

    print("\n--- Test Results ---")
    print(f"Total tokens received: {test.token_count}")
    print(f"Generation completed naturally: {test.streaming_complete}")

    assert test.token_count > 0, "No tokens were generated"
    assert test.streaming_complete, "Streaming did not complete naturally"


if __name__ == "__main__":
    print(f"Testing Ollama streaming at {OLLAMA_BASE_URL} with model {OLLAMA_LLM}")
    pytest.main(["-v", __file__])
