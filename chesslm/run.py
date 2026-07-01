import os
import sys

import torch

from . import config
from .inference import uci_loop, warmup_model
from .model import new_model
from .util import log
from .vocab import load_vocab


def run_inference():
    """Single-responsibility entry point for normal / engine usage.

    Requires MODEL_PATH and VOCAB_PATH to already exist on disk. It never
    trains and never downloads anything — if the artifacts are missing this
    is a hard error, pointing the user at `--train` or `--getmodel`.
    """
    if not (os.path.exists(config.MODEL_PATH) and os.path.exists(config.VOCAB_PATH)):
        log(
            f"ERROR: missing required files ({config.MODEL_PATH}, {config.VOCAB_PATH})."
        )
        log("Get a trained model first:")
        log("  ./chesSLM.py --getmodel   (download the pretrained model)")
        log("  ./chesSLM.py --train      (train one from scratch)")
        sys.exit(1)

    config.set_seed(config.SEED)

    vocab = load_vocab(config.VOCAB_PATH)
    raw_model = new_model(len(vocab))
    raw_model.load_state_dict(
        torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=True)
    )
    model = torch.compile(raw_model)
    params = sum(p.numel() for p in raw_model.parameters())
    log(f"- Vocab: {len(vocab):,}")
    log(f"- Model: {params:,} parameters")
    log(f"- Device: {config.DEVICE}\n")

    model.eval()
    warmup_model(model, vocab)
    log("ChesSLM ready — entering UCI loop")
    uci_loop(model, vocab)
