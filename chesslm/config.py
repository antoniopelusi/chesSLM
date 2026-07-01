import random

import numpy as np
import torch

from .paths import (  # noqa: F401  (re-exported for convenience)
    DATASET_ID,
    HF_REPO_ID,
    MODEL_PATH,
    S1_MODEL_PATH,
    S1_TRAIN_CACHE,
    S1_VAL_CACHE,
    S2_MODEL_PATH,
    S2_TRAIN_CACHE,
    S2_VAL_CACHE,
    VOCAB_PATH,
)

# ── architecture ──────────────────────────────────────────────────────────────
CONTEXT_LEN = 350
D_MODEL = 1024
N_HEADS = 16
N_LAYERS = 12
D_FF = 4096
DROPOUT = 0.15

# ── training (shared) ─────────────────────────────────────────────────────────
LOGICAL_BATCH = 1024
PHYSICAL_BATCH = 256
assert LOGICAL_BATCH % PHYSICAL_BATCH == 0
ACCUM_STEPS = LOGICAL_BATCH // PHYSICAL_BATCH
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.15
LOG_INTERVAL = 100
SEED = 42
MIN_LR = 1e-5

# ── validation / overfitting control ──────────────────────────────────────────
VAL_FRACTION = 0.03
EVAL_INTERVAL_STEPS = 150
EARLY_STOP_PATIENCE = 5

# ── stage 1 — pretraining: ELO in [S1_MIN_ELO, S2_MIN_ELO) ───────────────────
S1_MIN_ELO = 1800
S1_EPOCHS = 8
S1_LR = 4e-4
S1_WARMUP = 200

# ── stage 2 — finetuning: ELO >= S2_MIN_ELO ───────────────────────────────────
S2_MIN_ELO = 2400
S2_EPOCHS = 10
S2_LR = 8e-5
S2_WARMUP = 200

# ── inference ─────────────────────────────────────────────────────────────────
# Mutable at runtime (e.g. via UCI `setoption`) — other modules must read this
# through `config.TEMPERATURE`, never via `from .config import TEMPERATURE`,
# or they'll capture a stale copy instead of seeing live updates.
TEMPERATURE = 0.0

# ── device ────────────────────────────────────────────────────────────────────
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── special tokens ────────────────────────────────────────────────────────────
PAD_TOK = "<pad>"
BOS_TOK = "<bos>"
EOS_TOK = "<eos>"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
