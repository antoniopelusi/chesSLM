import json
import os
import random
from types import SimpleNamespace

import torch

DATASET_ID = "antoniopelusi/chesSLM-dataset"
MODEL_ID = "antoniopelusi/chesSLM"

VOCAB_PATH = "chesslm_vocab.json"
MODEL_PATH = "chesslm_model.safetensors"

PAD, BOS = 0, 1
SPECIAL_TOKENS = {"<pad>": PAD, "<bos>": BOS}

HEAD_DIM = 64

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "config.json")) as f:
    cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

assert cfg.model.d_model % HEAD_DIM == 0
assert 0 < cfg.data.val_fraction < 1
assert cfg.train.batch_tokens % cfg.train.micro_tokens == 0
assert cfg.train.micro_tokens >= cfg.model.context_len

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
