#!/usr/bin/env -S ./.venv/bin/python -u
"""
ChesSLM — entry point

Dispatches to one of three mutually exclusive modes based on CLI flags:

* ``(default)``  Load the trained model and enter the UCI inference loop.
* ``--train``    Train the model from scratch (or resume an interrupted run).
* ``--getmodel`` Download pretrained weights and vocabulary from Hugging Face.

All heavy imports (PyTorch, the chesslm package) are deferred until after GPU
selection and CLI parsing so that ``--getmodel`` never triggers a CUDA
initialisation and ``--help`` is instantaneous.
"""

import os
import shutil
import sys

# CUDA env must be set before any torch import
_cuda_lib = "/usr/local/cuda/lib64"
if _cuda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = (
        f"{_cuda_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}".strip(":")
    )
os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# GPU selection must happen before torch.device is evaluated in config
from chesslm.util import log, select_gpu

select_gpu()

# CLI parsing (lightweight — no torch needed for --getmodel)
import argparse

parser = argparse.ArgumentParser(prog="chesSLM", add_help=True)
group = parser.add_mutually_exclusive_group()
group.add_argument("--train", action="store_true", help="Train the model from scratch")
group.add_argument(
    "--getmodel",
    action="store_true",
    help="Download pretrained weights from Hugging Face",
)
args = parser.parse_args()


# --getmodel branch (no torch, no GPU needed)
if args.getmodel:
    from huggingface_hub import hf_hub_download

    from chesslm.paths import HF_REPO_ID, MODEL_PATH, VOCAB_PATH

    for filename, local_path in [(MODEL_PATH, MODEL_PATH), (VOCAB_PATH, VOCAB_PATH)]:
        if os.path.exists(local_path):
            log(f"Already exists, skipping: {local_path}")
            continue
        log(f"Downloading {filename} from {HF_REPO_ID} ...")
        downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
        # hf_hub_download caches to ~/.cache; copy to the working directory.
        shutil.copy(downloaded, local_path)
        log(f"Saved: {local_path}")

    log("Done. Run './chesSLM.py' to start the engine.")
    sys.exit(0)


# --train branch
if args.train:
    from chesslm.train import run_train

    run_train()
    sys.exit(0)


# Default branch: inference / UCI engine
from chesslm.run import run_inference

run_inference()
