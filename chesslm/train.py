import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import DATASET_ID, DEVICE, MODEL_PATH, PAD, VOCAB_PATH, cfg, set_seed
from .data import Pool, batches, load_data, plan
from .model import ChesSLM, save_checkpoint
from .util import log

ACCUM = cfg.train.batch_tokens // cfg.train.micro_tokens
GRAD_CLIP = 1.0
BETAS = (0.9, 0.95)
MIN_LR_RATIO = 0.1
LOG_INTERVAL = 100
EVALS_PER_EPOCH = 4
VOCAB_PAD = 64


def lm_loss(model, x, y):
    with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
        logits = model(x)
    return F.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        y.view(-1),
        ignore_index=PAD,
        reduction="sum",
    )


@torch.no_grad()
def evaluate(model, pool):
    total, count = torch.zeros((), device=DEVICE), 0
    for x, y in batches(pool, np.random.default_rng(0)):
        total += lm_loss(model, x, y)
        count += (y != PAD).sum()
    return (total / count).item()


def lr_at(step, total, warmup, lr, min_lr):
    if step <= warmup:
        return lr * step / warmup
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def status(tag, step, total, lr):
    gb = torch.cuda.max_memory_allocated() / 2**30
    return f"{tag} step {step}/{total} lr {lr:.2e} mem {gb:.1f} GB"


def run_stage(raw, model, name, train, val, stage):
    params = list(raw.parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in params if p.dim() >= 2],
                "weight_decay": cfg.train.weight_decay,
            },
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ],
        lr=stage.lr,
        betas=BETAS,
        fused=True,
    )
    steps_per_epoch = len(plan(train, np.random.default_rng(0))) // ACCUM
    total = stage.epochs * steps_per_epoch
    warmup = round(total * stage.warmup_ratio)
    min_lr = stage.lr * MIN_LR_RATIO
    log(
        f"{name}: {len(train.lens):,} games, {int((train.lens - 1).sum()):,} tokens,"
        f" {len(val.lens):,} validation games"
    )
    log(
        f"{name}: {stage.epochs} epochs x {steps_per_epoch:,} steps = {total:,} steps,"
        f" {warmup:,} warmup, lr {stage.lr:.0e}"
    )

    pick = np.random.default_rng(0).choice(
        len(train.lens), len(val.lens), replace=False
    )
    probe = Pool(train.tokens, train.starts[pick], train.lens[pick])
    eval_steps = {
        steps_per_epoch * k // EVALS_PER_EPOCH for k in range(1, EVALS_PER_EPOCH + 1)
    }

    rng = np.random.default_rng(cfg.train.seed)
    step = 0
    for epoch in range(1, stage.epochs + 1):
        tag = f"{name} ep {epoch}/{stage.epochs}"
        epoch_batches = batches(train, rng)
        running, n_run = torch.zeros(2, device=DEVICE), 0
        for i in range(1, steps_per_epoch + 1):
            step += 1
            lr = lr_at(step, total, warmup, stage.lr, min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            micro_batches = [next(epoch_batches) for _ in range(ACCUM)]
            n_tokens = sum((y != PAD).sum() for _, y in micro_batches)
            for x, y in micro_batches:
                loss = lm_loss(model, x, y) / n_tokens
                loss.backward()
                running[0] += loss.detach()
            n_run += 1
            running[1] += torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            if step % LOG_INTERVAL == 0:
                loss_avg, grad_avg = (running / n_run).tolist()
                log(
                    status(tag, step, total, lr)
                    + f" train_loss {loss_avg:.4f} grad {grad_avg:.2f}"
                )
                running.zero_()
                n_run = 0
            if i in eval_steps:
                log(
                    status(tag, step, total, lr)
                    + f" eval_train_loss {evaluate(raw, probe):.4f}"
                    f" eval_val_loss {evaluate(raw, val):.4f}"
                )


def run_train():
    set_seed(cfg.train.seed)
    log(f"Loading {DATASET_ID} ...")
    vocab, pools = load_data()
    vocab_size = -(-len(vocab) // VOCAB_PAD) * VOCAB_PAD
    raw = ChesSLM(vocab_size, **vars(cfg.model)).to(DEVICE)
    log(f"Vocab: {len(vocab):,} tokens (padded to {vocab_size:,})")
    log(f"Model: {sum(p.numel() for p in raw.parameters()):,} parameters")

    model = torch.compile(raw, dynamic=True)
    run_stage(raw, model, "stage1", pools["stage1"], pools["stage1_val"], cfg.stage1)
    run_stage(raw, model, "stage2", pools["stage2"], pools["stage2_val"], cfg.stage2)
    save_checkpoint(raw, MODEL_PATH)
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f)
    log(f"Training complete: {MODEL_PATH}, {VOCAB_PATH}")
