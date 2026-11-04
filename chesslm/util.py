import os
import subprocess
import sys
import time


def log(*args, **kwargs):
    print(time.strftime("%H:%M:%S"), *args, **kwargs, file=sys.stderr, flush=True)


def setup_cuda():
    lib = "/usr/local/cuda/lib64"
    paths = os.environ.get("LD_LIBRARY_PATH", "")
    if lib not in paths:
        os.environ["LD_LIBRARY_PATH"] = f"{lib}:{paths}".strip(":")
    os.environ["CUDA_HOME"] = "/usr/local/cuda"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return
    free = [int(x) for x in out.split()]
    best = free.index(max(free))
    log(f"GPU: {best} ({max(free)} MiB free)")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best)
