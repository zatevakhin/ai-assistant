import asyncio
import time

import pytest

from assistant.core import EventBus, service


class TestEventBusParallelExecution:
    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @service
    async def long_task(self, sleep_time, value):
        await asyncio.sleep(sleep_time)
        return f"{value}-{sleep_time}"

    @service
    async def failing_task(self):
        await asyncio.sleep(0.1)
        raise ValueError("Task failed")

    def test_parallel_execution_basic(self, event_bus):
        event_bus.register_service("test_plugin", "long_task", self.long_task)

        start_time = time.time()

        # Start 3 parallel tasks with 0.5s each
        futures = []
        for i in range(3):
            _, future = event_bus.call_service_async(
                "test_plugin", "long_task", 0.5, f"task{i}"
            )
            futures.append(future)

        # Get all results
        results = [f.result() for f in futures]

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Verify correct results
        assert results == ["task0-0.5", "task1-0.5", "task2-0.5"]

        # Verify parallel execution (should take ~0.5s, not ~1.5s)
        assert elapsed_time < 1.0, (
            f"Execution took {elapsed_time:.2f}s, suggesting sequential execution"
        )

    def test_tasks_different_durations(self, event_bus):
        event_bus.register_service("test_plugin", "long_task", self.long_task)

        # Start a long-running task (1.0s)
        start_time = time.time()
        _, long_future = event_bus.call_service_async(
            "test_plugin", "long_task", 1.0, "long"
        )

        # Small delay to ensure the first task starts first
        time.sleep(0.1)

        # Start a shorter task (0.3s) after the long one has started
        _, short_future = event_bus.call_service_async(
            "test_plugin", "long_task", 0.3, "short"
        )

        # The shorter task should finish first despite starting later
        short_result = short_future.result()
        long_result = long_future.result()

        end_time = time.time()
        elapsed_time = end_time - start_time

        # Verify results
        assert short_result == "short-0.3"
        assert long_result == "long-1.0"

        # Shorter task started later but should finish first
        # If tasks were truly parallel, total time should be just over 1.0s, not 1.3s+
        assert elapsed_time < 1.2, f"Execution took {elapsed_time:.2f}s"
        assert elapsed_time > 1.0, (
            f"Long task completed too quickly: {elapsed_time:.2f}s"
        )

    def test_error_handling(self, event_bus):
        event_bus.register_service("test_plugin", "failing_task", self.failing_task)

        _, future = event_bus.call_service_async("test_plugin", "failing_task")

        with pytest.raises(ValueError, match="Task failed"):
            future.result()

    def test_cancel_task(self, event_bus):
        event_bus.register_service("test_plugin", "long_task", self.long_task)

        request_id, future = event_bus.call_service_async(
            "test_plugin", "long_task", 10.0, "very_long"
        )

        # Cancel the task
        canceled = event_bus.cancel_call(request_id)
        assert canceled
        assert future.cancelled()
