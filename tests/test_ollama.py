import ollama
import time
import concurrent.futures
import os
import pytest

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM = os.getenv("OLLAMA_LLM", "deepseek-r1:1.5b")


def generate_response(prompt, model=OLLAMA_LLM):
    try:
        client = ollama.Client(host=OLLAMA_BASE_URL)
        response = client.chat(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return {"success": True, "response": response["message"]["content"]}
    except Exception as e:
        return {"success": False, "error": str(e)}


def test_single_request():
    result = generate_response("What is machine learning?")
    print(result)
    assert result["success"], "Single request failed"
    assert len(result["response"]) > 0, "Empty response received"


def test_parallel_requests():
    num_requests = 3
    prompts = [
        f"Write a short definition of AI (request {i})" for i in range(num_requests)
    ]

    print(
        f"\nRunning {num_requests} parallel requests to {OLLAMA_BASE_URL} using {OLLAMA_LLM}"
    )
    start_time = time.time()

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(generate_response, prompt) for prompt in prompts]

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    total_time = time.time() - start_time

    successful = sum(1 for r in results if r["success"])

    print(f"Successful requests: {successful}/{num_requests}")
    print(f"Total time: {total_time:.2f} seconds")

    assert successful == num_requests


if __name__ == "__main__":
    print(f"Testing Ollama at {OLLAMA_BASE_URL} with model {OLLAMA_LLM}")
    pytest.main(["-v", __file__])
