import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import subprocess
import sys


def log(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)


def select_gpu():
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


select_gpu()

import json
import math
import random

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

# ── paths ─────────────────────────────────────────────────────────────────────
DATASET_ID = "angeluriot/chess_games"
VOCAB_PATH = "chesslm_vocab.json"
MODEL_PATH = "chesslm_model.pt"

S1_TRAIN_CACHE = "chesslm_data_s1_train.pt"
S1_VAL_CACHE = "chesslm_data_s1_val.pt"
S2_TRAIN_CACHE = "chesslm_data_s2_train.pt"
S2_VAL_CACHE = "chesslm_data_s2_val.pt"

S1_MODEL_PATH = "chesslm_stage1_best.pt"
S2_MODEL_PATH = "chesslm_stage2_best.pt"

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


# ── data ──────────────────────────────────────────────────────────────────────


def download_games(min_elo):
    log(f"Streaming dataset: {DATASET_ID} (ELO >= {min_elo}) ...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    games = []
    scanned = 0
    kept = 0

    for sample in ds:
        scanned += 1
        white_elo = sample.get("white_elo")
        black_elo = sample.get("black_elo")
        if white_elo is None or black_elo is None:
            continue
        try:
            elo = min(int(white_elo), int(black_elo))
        except (TypeError, ValueError):
            continue
        if elo < min_elo:
            continue
        moves = [m.rstrip("+#") for m in sample.get("moves_san", [])]
        if len(moves) >= 10:
            games.append((elo, moves))
            kept += 1
        if scanned % 100_000 == 0:
            log(f"  Scanned {scanned:,} — kept {kept:,} games ...")

    log(f"Collected {kept:,} valid games from {scanned:,} samples")
    return games


def split_games(games, val_fraction, seed):
    if len(games) < 20:
        return games, []
    rng = random.Random(seed)
    order = list(range(len(games)))
    rng.shuffle(order)
    n_val = max(1, round(len(games) * val_fraction))
    val_idx = set(order[:n_val])
    train = [g for i, g in enumerate(games) if i not in val_idx]
    val = [g for i, g in enumerate(games) if i in val_idx]
    return train, val


def build_vocab(games):
    tokens = {PAD_TOK, BOS_TOK, EOS_TOK}
    for game in games:
        tokens.update(game)
    return {t: i for i, t in enumerate(sorted(tokens))}


def save_vocab(vocab, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(vocab, f)
    os.replace(tmp, path)


def load_vocab(path):
    with open(path) as f:
        return json.load(f)


def build_token_stream(games, vocab):
    bos = vocab[BOS_TOK]
    eos = vocab[EOS_TOK]
    stream = []
    for g in games:
        stream.append(bos)
        for m in g:
            tok = vocab.get(m)
            if tok is not None:
                stream.append(tok)
        stream.append(eos)
    return torch.tensor(stream, dtype=torch.long)


def _atomic_tensor_save(tensor, path):
    tmp = path + ".tmp"
    torch.save(tensor, tmp)
    os.replace(tmp, path)


class ChessDataset(Dataset):
    def __init__(self, data, ctx):
        n = (len(data) - 1) // ctx
        self.data = data[: n * ctx + 1]
        self.ctx = ctx
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = idx * self.ctx
        chunk = self.data[s : s + self.ctx + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


def make_loader(data, batch_size, train):
    dataset = ChessDataset(data, CONTEXT_LEN)
    if len(dataset) == 0:
        return None
    n_workers = min(4, os.cpu_count() or 1)
    g = torch.Generator()
    g.manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=n_workers,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(n_workers > 0),
        drop_last=train,
        generator=g if train else None,
    )


# ── model ─────────────────────────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=True,
        )
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class ChesSLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, ctx_len, dropout):
        super().__init__()
        self.ctx = ctx_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if m is self.head:
                continue
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln_f(h))


# ── training ──────────────────────────────────────────────────────────────────


def cosine_lr(step, total, warmup, max_lr, min_lr=MIN_LR):
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))


def _make_amp_config():
    if DEVICE.type != "cuda":
        return False, torch.float32, None
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, None
    return True, torch.float16, torch.amp.GradScaler("cuda")


def _make_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() <= 1 or "bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _step_optimizer(params, optimizer, scaler, opt_step, total_steps, max_lr, warmup):
    opt_step += 1
    lr = cosine_lr(opt_step, total_steps, warmup, max_lr)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    if scaler is not None:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
    else:
        nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return opt_step, lr


@torch.inference_mode()
def evaluate(model, loader, pad_id):
    if loader is None or len(loader) == 0:
        return float("nan")
    use_amp, amp_dtype, _ = _make_amp_config()
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V), y.view(B * T), ignore_index=pad_id
            )
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def train_model(
    model,
    raw_model,
    train_loader,
    val_loader,
    vocab,
    epochs,
    total_steps,
    max_lr,
    warmup,
    best_ckpt_path,
    eval_interval=EVAL_INTERVAL_STEPS,
    patience=EARLY_STOP_PATIENCE,
):
    pad_id = vocab[PAD_TOK]
    use_amp, amp_dtype, scaler = _make_amp_config()
    optimizer = torch.optim.AdamW(
        _make_param_groups(raw_model, WEIGHT_DECAY),
        lr=max_lr,
        betas=(0.9, 0.95),
    )
    all_params = list(raw_model.parameters())
    has_val = val_loader is not None and len(val_loader) > 0
    best_val = float("inf")
    no_improve = 0
    opt_step = 0
    lr = 0.0
    stop = False

    def check_val(tag):
        nonlocal best_val, no_improve, stop
        val_loss = evaluate(model, val_loader, pad_id)
        improved = val_loss < best_val - 1e-4
        if improved:
            best_val = val_loss
            no_improve = 0
            tmp = best_ckpt_path + ".tmp"
            torch.save(raw_model.state_dict(), tmp)
            os.replace(tmp, best_ckpt_path)
        else:
            no_improve += 1
        log(f"{tag}  val_loss={val_loss:.4f}{'  * new best' if improved else ''}")
        if no_improve >= patience:
            log(f"Early stopping: no improvement for {patience} evaluations")
            stop = True

    for epoch in range(1, epochs + 1):
        if stop:
            break
        model.train()
        micro_step = 0
        running, running_n = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            micro_step += 1

            with autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.view(B * T, V), y.view(B * T), ignore_index=pad_id
                )

            scaled = loss / ACCUM_STEPS
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            running += loss.item()
            running_n += 1

            if micro_step % ACCUM_STEPS != 0:
                continue

            opt_step, lr = _step_optimizer(
                all_params, optimizer, scaler, opt_step, total_steps, max_lr, warmup
            )

            if opt_step % LOG_INTERVAL == 0:
                log(
                    f"[step {opt_step:>6}/{total_steps}]"
                    f"  loss={running / running_n:.4f}  lr={lr:.2e}"
                )
                running, running_n = 0.0, 0

            if has_val and opt_step % eval_interval == 0:
                check_val(f"[step {opt_step:>6}/{total_steps}]")
                if stop:
                    break

        if stop:
            break

        if micro_step % ACCUM_STEPS != 0:
            opt_step, lr = _step_optimizer(
                all_params, optimizer, scaler, opt_step, total_steps, max_lr, warmup
            )

        if running_n > 0:
            log(
                f"[step {opt_step:>6}/{total_steps}]"
                f"  loss={running / running_n:.4f}  lr={lr:.2e}"
            )

        if has_val:
            check_val(f"Epoch {epoch}/{epochs}")
        else:
            log(f"Epoch {epoch}/{epochs} complete")

        if stop:
            break

    if has_val and os.path.exists(best_ckpt_path):
        raw_model.load_state_dict(
            torch.load(best_ckpt_path, map_location=DEVICE, weights_only=True)
        )
        log(f"Restored best checkpoint (val_loss={best_val:.4f}) from {best_ckpt_path}")
    elif not has_val:
        tmp = best_ckpt_path + ".tmp"
        torch.save(raw_model.state_dict(), tmp)
        os.replace(tmp, best_ckpt_path)
        log(f"No validation set — saved final weights to {best_ckpt_path}")


# ── inference ─────────────────────────────────────────────────────────────────


def get_best_move(model, vocab, board, game_moves):
    pad_id = vocab[PAD_TOK]
    ids = [vocab[BOS_TOK]] + [vocab.get(m, pad_id) for m in game_moves]
    ids = ids[-CONTEXT_LEN:]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    with torch.inference_mode():
        logits = model(x)[0, -1]

    legal_map = {board.san(m).rstrip("+#"): m for m in board.legal_moves}
    legal_tokens = [
        (san, vocab[san], move) for san, move in legal_map.items() if san in vocab
    ]

    if not legal_tokens:
        return random.choice(list(board.legal_moves)).uci()

    tids = torch.tensor([tid for _, tid, _ in legal_tokens], device=DEVICE)
    scores = logits[tids]

    if TEMPERATURE <= 0.0:
        idx = scores.argmax().item()
    else:
        probs = F.softmax(scores / TEMPERATURE, dim=0)
        idx = torch.multinomial(probs, 1).item()

    return legal_tokens[idx][2].uci()


def uci_loop(model, vocab):
    board = chess.Board()
    game_moves = []

    for line in sys.stdin:
        cmd = line.strip()
        if not cmd:
            continue

        if cmd == "uci":
            print("id name ChesSLM")
            print("id author ChesSLM")
            print("uciok")

        elif cmd == "isready":
            print("readyok")

        elif cmd == "ucinewgame":
            board = chess.Board()
            game_moves = []

        elif cmd == "stop":
            pass

        elif cmd.startswith("position"):
            board = chess.Board()
            game_moves = []
            parts = cmd.split()
            i = 1

            if i < len(parts) and parts[i] == "startpos":
                i += 1
            elif i < len(parts) and parts[i] == "fen":
                fen = " ".join(parts[i + 1 : i + 7])
                try:
                    board = chess.Board(fen)
                except ValueError:
                    board = chess.Board()
                i += 7

            if i < len(parts) and parts[i] == "moves":
                for uci_m in parts[i + 1 :]:
                    try:
                        m = chess.Move.from_uci(uci_m)
                        game_moves.append(board.san(m).rstrip("+#"))
                        board.push(m)
                    except Exception:
                        break

        elif cmd.startswith("go"):
            mv = get_best_move(model, vocab, board, game_moves)
            print(f"bestmove {mv}")

        elif cmd == "quit":
            break

        sys.stdout.flush()


# ── helpers ───────────────────────────────────────────────────────────────────


def _new_model(vocab_size):
    return ChesSLM(
        vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF, CONTEXT_LEN, DROPOUT
    ).to(DEVICE)


def _log_stage(tag, train_loader, val_loader, total_steps, warmup):
    val_chunks = len(val_loader.dataset) if val_loader is not None else 0
    log(
        f"[{tag}]"
        f"  train_chunks={len(train_loader.dataset):,}"
        f"  val_chunks={val_chunks:,}"
        f"  batches/epoch={len(train_loader):,}"
        f"  steps={total_steps:,}"
        f"  warmup={warmup}"
        f"  batch={LOGICAL_BATCH}"
        f"  accum={ACCUM_STEPS}x"
    )


def load_or_build_data():
    caches = [VOCAB_PATH, S1_TRAIN_CACHE, S1_VAL_CACHE, S2_TRAIN_CACHE, S2_VAL_CACHE]
    if all(os.path.exists(p) for p in caches):
        log("Data cache found — loading ...")
        vocab = load_vocab(VOCAB_PATH)
        return (
            vocab,
            torch.load(S1_TRAIN_CACHE, weights_only=True),
            torch.load(S1_VAL_CACHE, weights_only=True),
            torch.load(S2_TRAIN_CACHE, weights_only=True),
            torch.load(S2_VAL_CACHE, weights_only=True),
        )

    log("Downloading data (single pass over the dataset) ...")
    pool = download_games(S1_MIN_ELO)
    s1_pool = [moves for elo, moves in pool if elo < S2_MIN_ELO]
    s2_pool = [moves for elo, moves in pool if elo >= S2_MIN_ELO]
    log(f"Stage 1 pool [{S1_MIN_ELO}, {S2_MIN_ELO}): {len(s1_pool):,} games")
    log(f"Stage 2 pool >= {S2_MIN_ELO}: {len(s2_pool):,} games")

    s1_train_g, s1_val_g = split_games(s1_pool, VAL_FRACTION, SEED)
    s2_train_g, s2_val_g = split_games(s2_pool, VAL_FRACTION, SEED + 1)
    log(f"Stage 1 split — train: {len(s1_train_g):,}  val: {len(s1_val_g):,}")
    log(f"Stage 2 split — train: {len(s2_train_g):,}  val: {len(s2_val_g):,}")

    vocab = build_vocab(s1_train_g + s1_val_g + s2_train_g + s2_val_g)
    save_vocab(vocab, VOCAB_PATH)
    log(f"Vocab: {len(vocab)} tokens")

    s1_train = build_token_stream(s1_train_g, vocab)
    _atomic_tensor_save(s1_train, S1_TRAIN_CACHE)
    s1_val = build_token_stream(s1_val_g, vocab)
    _atomic_tensor_save(s1_val, S1_VAL_CACHE)
    s2_train = build_token_stream(s2_train_g, vocab)
    _atomic_tensor_save(s2_train, S2_TRAIN_CACHE)
    s2_val = build_token_stream(s2_val_g, vocab)
    _atomic_tensor_save(s2_val, S2_VAL_CACHE)
    log("Data cache saved")

    return vocab, s1_train, s1_val, s2_train, s2_val


# ── entry point ───────────────────────────────────────────────────────────────


def main():
    set_seed(SEED)

    # ── final model already exists ────────────────────────────────────────────
    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        vocab = load_vocab(VOCAB_PATH)
        raw_model = _new_model(len(vocab))
        raw_model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
        model = torch.compile(raw_model)
        params = sum(p.numel() for p in raw_model.parameters())
        log(f"- Vocab: {len(vocab):,}")
        log(f"- tokens Model: {params:,} parameters")
        log(f"- Device: {DEVICE}\n")
        model.eval()
        log("ChesSLM ready — entering UCI loop")
        uci_loop(model, vocab)
        return

    vocab, s1_train, s1_val, s2_train, s2_val = load_or_build_data()

    raw_model = _new_model(len(vocab))

    log("Compiling ...")
    model = torch.compile(raw_model)

    # ── stage 1 — pretraining ─────────────────────────────────────────────────
    if os.path.exists(S1_MODEL_PATH):
        log("Stage 1 checkpoint found — skipping pretraining ...")
        raw_model.load_state_dict(
            torch.load(S1_MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
    else:
        s1_train_loader = make_loader(s1_train, PHYSICAL_BATCH, train=True)
        s1_val_loader = make_loader(s1_val, PHYSICAL_BATCH, train=False)
        s1_steps = S1_EPOCHS * math.ceil(len(s1_train_loader) / ACCUM_STEPS)
        if s1_steps < S1_WARMUP:
            log(f"WARNING: stage 1 total_steps ({s1_steps}) < S1_WARMUP ({S1_WARMUP})")
        _log_stage(
            "stage 1 / pretraining", s1_train_loader, s1_val_loader, s1_steps, S1_WARMUP
        )
        train_model(
            model,
            raw_model,
            s1_train_loader,
            s1_val_loader,
            vocab,
            S1_EPOCHS,
            s1_steps,
            S1_LR,
            S1_WARMUP,
            S1_MODEL_PATH,
        )
        log(f"Stage 1 best checkpoint: {S1_MODEL_PATH}")

    # ── stage 2 — finetuning ──────────────────────────────────────────────────
    if os.path.exists(S2_MODEL_PATH):
        log("Stage 2 checkpoint found — skipping finetuning ...")
        raw_model.load_state_dict(
            torch.load(S2_MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
    else:
        s2_train_loader = make_loader(s2_train, PHYSICAL_BATCH, train=True)
        s2_val_loader = make_loader(s2_val, PHYSICAL_BATCH, train=False)
        s2_steps = S2_EPOCHS * math.ceil(len(s2_train_loader) / ACCUM_STEPS)
        if s2_steps < S2_WARMUP:
            log(f"WARNING: stage 2 total_steps ({s2_steps}) < S2_WARMUP ({S2_WARMUP})")
        _log_stage(
            "stage 2 / finetuning", s2_train_loader, s2_val_loader, s2_steps, S2_WARMUP
        )
        train_model(
            model,
            raw_model,
            s2_train_loader,
            s2_val_loader,
            vocab,
            S2_EPOCHS,
            s2_steps,
            S2_LR,
            S2_WARMUP,
            S2_MODEL_PATH,
        )
        log(f"Stage 2 best checkpoint: {S2_MODEL_PATH}")

    tmp = MODEL_PATH + ".tmp"
    torch.save(raw_model.state_dict(), tmp)
    os.replace(tmp, MODEL_PATH)
    log(f"Final model saved: {MODEL_PATH}")

    model.eval()
    log("ChesSLM ready — entering UCI loop")
    uci_loop(model, vocab)


if __name__ == "__main__":
    main()
