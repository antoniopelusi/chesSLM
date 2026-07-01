import os
import subprocess
import sys


def log(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def select_gpu():
    """Pick the GPU with the most free memory and pin CUDA_VISIBLE_DEVICES to it.

    No-op (silently) if nvidia-smi is unavailable or fails — CPU/single-GPU
    setups are unaffected.
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
