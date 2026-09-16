"""
performance.py
---------------
Latency and throughput (FPS) measurement utilities for the CPU
software baseline. These numbers are what will later be compared
directly against the FPGA hardware measurements.
"""

import time
import statistics


class Timer:
    """Simple context-manager timer for measuring elapsed wall-clock time.

    Example:
        with Timer() as t:
            do_work()
        print(t.elapsed_ms)
    """

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info):
        self._end = time.perf_counter()
        self.elapsed_seconds = self._end - self._start
        self.elapsed_ms = self.elapsed_seconds * 1000.0


def measure_detection_latency(detect_fn, *args, repeats=10, **kwargs):
    """Measure the latency of a detection call over multiple repeats.

    Args:
        detect_fn (callable): Function performing detection, e.g. detect_faces.
        *args: Positional args passed to detect_fn.
        repeats (int): Number of times to repeat the call for stable timing.
        **kwargs: Keyword args passed to detect_fn.

    Returns:
        dict: {
            'mean_ms': float, 'min_ms': float, 'max_ms': float,
            'stdev_ms': float, 'all_ms': list[float], 'result': last call's output
        }
    """
    timings_ms = []
    result = None

    for _ in range(repeats):
        with Timer() as t:
            result = detect_fn(*args, **kwargs)
        timings_ms.append(t.elapsed_ms)

    return {
        "mean_ms": statistics.mean(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "stdev_ms": statistics.pstdev(timings_ms) if len(timings_ms) > 1 else 0.0,
        "all_ms": timings_ms,
        "result": result,
    }


def compute_fps(mean_latency_ms):
    """Convert a mean per-image latency (ms) into frames-per-second.

    Args:
        mean_latency_ms (float): Mean latency in milliseconds.

    Returns:
        float: Frames per second (0 if latency is 0).
    """
    if mean_latency_ms <= 0:
        return 0.0
    return 1000.0 / mean_latency_ms


def summarize_dataset_performance(per_image_latencies_ms):
    """Summarize latency/throughput across an entire dataset.

    Args:
        per_image_latencies_ms (dict): filename -> latency in ms.

    Returns:
        dict: {
            'total_time_ms': float,
            'mean_latency_ms': float,
            'fps': float,
            'num_images': int,
        }
    """
    values = list(per_image_latencies_ms.values())
    total_time_ms = sum(values)
    mean_latency_ms = statistics.mean(values) if values else 0.0

    return {
        "total_time_ms": total_time_ms,
        "mean_latency_ms": mean_latency_ms,
        "fps": compute_fps(mean_latency_ms),
        "num_images": len(values),
    }
