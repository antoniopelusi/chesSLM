import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from . import config
from .data import load_or_build_data, make_loader
from .model import new_model
from .util import log


def cosine_lr(step, total, warmup, max_lr, min_lr=None):
    min_lr = config.MIN_LR if min_lr is None else min_lr
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))


def _make_amp_config():
    if config.DEVICE.type != "cuda":
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
        nn.utils.clip_grad_norm_(params, config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
    else:
        nn.utils.clip_grad_norm_(params, config.GRAD_CLIP)
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
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        with autocast(device_type=config.DEVICE.type, dtype=amp_dtype, enabled=use_amp):
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
    eval_interval=None,
    patience=None,
):
    eval_interval = (
        config.EVAL_INTERVAL_STEPS if eval_interval is None else eval_interval
    )
    patience = config.EARLY_STOP_PATIENCE if patience is None else patience

    pad_id = vocab[config.PAD_TOK]
    use_amp, amp_dtype, scaler = _make_amp_config()
    optimizer = torch.optim.AdamW(
        _make_param_groups(raw_model, config.WEIGHT_DECAY),
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
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            micro_step += 1

            with autocast(
                device_type=config.DEVICE.type, dtype=amp_dtype, enabled=use_amp
            ):
                logits = model(x)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.view(B * T, V), y.view(B * T), ignore_index=pad_id
                )

            scaled = loss / config.ACCUM_STEPS
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            running += loss.item()
            running_n += 1

            if micro_step % config.ACCUM_STEPS != 0:
                continue

            opt_step, lr = _step_optimizer(
                all_params, optimizer, scaler, opt_step, total_steps, max_lr, warmup
            )

            if opt_step % config.LOG_INTERVAL == 0:
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

        if micro_step % config.ACCUM_STEPS != 0:
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
            torch.load(best_ckpt_path, map_location=config.DEVICE, weights_only=True)
        )
        log(f"Restored best checkpoint (val_loss={best_val:.4f}) from {best_ckpt_path}")
    elif not has_val:
        tmp = best_ckpt_path + ".tmp"
        torch.save(raw_model.state_dict(), tmp)
        os.replace(tmp, best_ckpt_path)
        log(f"No validation set — saved final weights to {best_ckpt_path}")


def _log_stage(tag, train_loader, val_loader, total_steps, warmup):
    val_chunks = len(val_loader.dataset) if val_loader is not None else 0
    log(
        f"[{tag}]"
        f"  train_chunks={len(train_loader.dataset):,}"
        f"  val_chunks={val_chunks:,}"
        f"  batches/epoch={len(train_loader):,}"
        f"  steps={total_steps:,}"
        f"  warmup={warmup}"
        f"  batch={config.LOGICAL_BATCH}"
        f"  accum={config.ACCUM_STEPS}x"
    )


def _run_stage(
    tag, model, raw_model, train_data, val_data, vocab, epochs, lr, warmup, ckpt_path
):
    if os.path.exists(ckpt_path):
        log(f"[{tag}] checkpoint found — skipping ...")
        raw_model.load_state_dict(
            torch.load(ckpt_path, map_location=config.DEVICE, weights_only=True)
        )
        return

    train_loader = make_loader(train_data, config.PHYSICAL_BATCH, train=True)
    val_loader = make_loader(val_data, config.PHYSICAL_BATCH, train=False)
    total_steps = epochs * math.ceil(len(train_loader) / config.ACCUM_STEPS)
    if total_steps < warmup:
        log(f"WARNING: [{tag}] total_steps ({total_steps}) < warmup ({warmup})")
    _log_stage(tag, train_loader, val_loader, total_steps, warmup)
    train_model(
        model,
        raw_model,
        train_loader,
        val_loader,
        vocab,
        epochs,
        total_steps,
        lr,
        warmup,
        ckpt_path,
    )
    log(f"[{tag}] best checkpoint: {ckpt_path}")


def run_train():
    """Single-responsibility entry point for the one-off training flow.

    Refuses to run if a final trained model is already present (training is
    meant to be a once-and-done operation) — remove the artifacts first if a
    retrain is really wanted. Intermediate per-stage checkpoints are still
    reused to allow resuming an interrupted run.
    """
    if os.path.exists(config.MODEL_PATH) or os.path.exists(config.VOCAB_PATH):
        log(f"ERROR: {config.MODEL_PATH} and/or {config.VOCAB_PATH} already exist.")
        log(
            "Training is a one-off operation. Remove these files first if you really want to retrain."
        )
        sys.exit(1)

    config.set_seed(config.SEED)
    vocab, s1_train, s1_val, s2_train, s2_val = load_or_build_data()

    raw_model = new_model(len(vocab))
    log("Compiling ...")
    model = torch.compile(raw_model)

    _run_stage(
        "stage 1 / pretraining",
        model,
        raw_model,
        s1_train,
        s1_val,
        vocab,
        config.S1_EPOCHS,
        config.S1_LR,
        config.S1_WARMUP,
        config.S1_MODEL_PATH,
    )
    _run_stage(
        "stage 2 / finetuning",
        model,
        raw_model,
        s2_train,
        s2_val,
        vocab,
        config.S2_EPOCHS,
        config.S2_LR,
        config.S2_WARMUP,
        config.S2_MODEL_PATH,
    )

    tmp = config.MODEL_PATH + ".tmp"
    torch.save(raw_model.state_dict(), tmp)
    os.replace(tmp, config.MODEL_PATH)
    log(f"Final model saved: {config.MODEL_PATH}")
    log("Training complete. Run './chesSLM.py' to use the model.")
