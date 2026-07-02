import os
import subprocess
import sys


def log(*args, **kwargs):
    """Write a message to ``stderr`` and flush immediately.

    All positional and keyword arguments are forwarded to :func:`print`, with
    ``file=sys.stderr`` and ``flush=True`` always set. This keeps engine
    diagnostics on ``stderr`` separate from the UCI protocol messages sent to
    ``stdout``, which is a requirement of the UCI specification.

    Args:
        *args: Values to print, joined by the separator (default space).
        **kwargs: Additional keyword arguments forwarded to :func:`print`
            (e.g. ``sep``, ``end``). ``file`` and ``flush`` are overridden.
    """
    print(*args, **kwargs, file=sys.stderr, flush=True)


def select_gpu():
    """Pin the process to the GPU with the most free VRAM via ``CUDA_VISIBLE_DEVICES``.

    Queries ``nvidia-smi`` for per-GPU free memory, selects the GPU with the
    highest value, and sets ``CUDA_VISIBLE_DEVICES`` to its index so that
    PyTorch sees only that device as ``cuda:0``.

    This must be called *before* any ``import torch`` statement so that the
    environment variable is in place when PyTorch initialises the CUDA context.

    The function is a no-op — and does not raise — when ``nvidia-smi`` is
    unavailable or returns unexpected output. CPU-only and single-GPU setups
    are therefore unaffected.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        free = [int(x) for x in result.stdout.strip().split("\n") if x.strip()]
        if not free:
            return
        best = free.index(max(free))
        log(f"- GPU {best} selected ({max(free)} MiB free)")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best)
    except Exception:
        pass
