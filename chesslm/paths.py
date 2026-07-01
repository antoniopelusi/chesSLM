# ── paths / remote sources ──────────────────────────────────────────────────
# Intentionally dependency-free (no torch/chess imports) so that
# `--getmodel` stays a lightweight, fast one-off download.

DATASET_ID = "angeluriot/chess_games"
HF_REPO_ID = "antoniopelusi/chesSLM"

VOCAB_PATH = "chesslm_vocab.json"
MODEL_PATH = "chesslm_model.pt"

S1_TRAIN_CACHE = "chesslm_data_s1_train.pt"
S1_VAL_CACHE = "chesslm_data_s1_val.pt"
S2_TRAIN_CACHE = "chesslm_data_s2_train.pt"
S2_VAL_CACHE = "chesslm_data_s2_val.pt"

S1_MODEL_PATH = "chesslm_stage1_best.pt"
S2_MODEL_PATH = "chesslm_stage2_best.pt"
