"""Centralized torch configuration for thread-safe operation."""

import contextlib
import os
import threading


class _TorchConfig:
    """Singleton for torch configuration state."""

    configured = False
    lock = threading.Lock()


def configure_torch_env() -> None:
    """Set environment variables BEFORE torch import (call at module level)."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


def configure_torch_runtime() -> None:
    """Configure torch AFTER import (call when analysis starts)."""
    with _TorchConfig.lock:
        if _TorchConfig.configured:
            return

        import torch

        torch.set_num_threads(1)

        with contextlib.suppress(RuntimeError):
            torch.multiprocessing.set_start_method("fork", force=True)

        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)

        _TorchConfig.configured = True
