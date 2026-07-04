import random

import numpy as np
import torch

from .paths import (
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
# Sized to balance representational capacity against the fresh-token budget
# produced by the ELO routing below (stage 1 has no rating floor, so its
# pool is large — see S1_EPOCHS for the resulting epoch budget). RoPE (see
# model.py) supplies positional information without a learned embedding
# table, at negligible parameter cost and with better generalisation to
# under-represented context lengths.
CONTEXT_LEN = 350
D_MODEL = 896
N_HEADS = 14
N_LAYERS = 10
D_FF = 3584
# Stage pools are each replayed for several epochs (see S1_EPOCHS/S2_EPOCHS),
# so this sits in the moderate range appropriate for repeated-epoch training
# on a fixed pool, rather than the lower values typical of single-pass
# training on unique data.
DROPOUT = 0.10

# ── training (shared) ─────────────────────────────────────────────────────────
LOGICAL_BATCH = 1024
PHYSICAL_BATCH = 256
assert LOGICAL_BATCH % PHYSICAL_BATCH == 0
ACCUM_STEPS = LOGICAL_BATCH // PHYSICAL_BATCH
GRAD_CLIP = 1.0
# Applied to all 2D+ parameters, including the tied embedding/output matrix
# (see train.py:_make_param_groups). Matched to DROPOUT's rationale: moderate
# regularisation for a model trained over several epochs of a fixed pool.
WEIGHT_DECAY = 0.10
LOG_INTERVAL = 100
SEED = 42
MIN_LR = 1e-5

# ── validation / overfitting control ──────────────────────────────────────────
# With stage-1/stage-2 pools in the millions of games, a small fraction
# still yields well over 100k validation games per stage without spending
# an unnecessarily large share of the data on validation.
VAL_FRACTION = 0.015
EVAL_INTERVAL_STEPS = 150
EARLY_STOP_PATIENCE = 5

# ── stage 1 — pretraining: everything not routed to stage 2 ──────────────────
# No lower ELO bound: a game only needs a *known* rating on either side >=
# S2_MIN_ELO to go to stage 2 (see data.download_games). Everything else —
# unrated, partially rated, or rated below S2_MIN_ELO — goes to stage 1.
# Expected pool size: ~8M games.
#
# EPOCHS: a ceiling, not a target — early stopping is expected to cut
# training short once validation loss plateaus.
S1_EPOCHS = 5
S1_LR = 4e-4
S1_WARMUP = 200

# ── stage 2 — finetuning: at least one player with a known ELO >= S2_MIN_ELO ─
# Expected pool size: ~6M games.
S2_MIN_ELO = 2400
S2_EPOCHS = 6
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
