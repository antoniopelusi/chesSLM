#!/usr/bin/env -S ./.venv/bin/python -u
import argparse
import os

from chesslm.util import setup_cuda

setup_cuda()

import torch

from chesslm.config import MODEL_PATH, VOCAB_PATH

parser = argparse.ArgumentParser(prog="chesSLM")
group = parser.add_mutually_exclusive_group()
group.add_argument("--train", action="store_true", help="Train the model from scratch")
group.add_argument("--getmodel", action="store_true", help="Download the trained model")
args = parser.parse_args()

if args.train:
    if os.path.exists(MODEL_PATH):
        parser.error(f"{MODEL_PATH} already exists. Remove it to retrain.")
    if not torch.cuda.is_available():
        parser.error("Training requires a CUDA GPU.")
    from chesslm.train import run_train

    run_train()

elif args.getmodel:
    from chesslm.getmodel import run_getmodel

    run_getmodel()
else:
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH)):
        parser.error(
            f"Missing {MODEL_PATH} or {VOCAB_PATH}. Run --getmodel or --train."
        )
    from chesslm.inference import run_inference

    run_inference()
