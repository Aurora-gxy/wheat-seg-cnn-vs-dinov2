"""Small helpers: reproducibility, device selection, timing."""
import random
import time

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds so experiments are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return the best available torch device: CUDA > Apple MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device=None) -> None:
    """Wait for queued GPU work to finish, so timings are accurate."""
    try:
        import torch
    except ImportError:
        return
    if device is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    """Usage:  with Timer() as t: ...;  t.seconds"""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start
        return False
