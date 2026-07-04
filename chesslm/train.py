import math
import os

import torch
import torch.nn.functional as F

from . import config
from .data import load_or_build_data, make_loader
from .model import new_model
from .util import log


def cosine_lr(step, total_steps, warmup_steps, max_lr, min_lr):
    """Compute the learning rate at *step* under a linear-warmup, cosine-decay schedule.

    Args:
        step (int): Current optimizer step (1-indexed).
        total_steps (int): Total number of optimizer steps for this stage.
        warmup_steps (int): Number of linear warmup steps.
        max_lr (float): Peak learning rate reached at the end of warmup.
        min_lr (float): Floor learning rate at the end of the cosine decay.

    Returns:
        float: Learning rate for *step*.
    """
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _make_amp_config():
    """Select the best available autocast dtype and matching GradScaler for this device.

    Prefers ``bfloat16`` on CUDA devices that support it (no scaler needed,
    since bf16's exponent range matches fp32 closely enough to avoid
    underflow). Falls back to ``float16`` with an active
    :class:`torch.cuda.amp.GradScaler` on older CUDA devices. AMP is disabled
    entirely on CPU.

    Returns:
        tuple[torch.dtype | None, torch.cuda.amp.GradScaler | None]:
            ``(autocast_dtype, scaler)``. Both are ``None`` on CPU.
    """
    if config.DEVICE.type != "cuda":
        return None, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    return torch.float16, torch.cuda.amp.GradScaler()


def _make_param_groups(model):
    """Split model parameters into decayed and non-decayed optimizer groups.

    Parameters with fewer than 2 dimensions (biases, LayerNorm weight/gain)
    are exempt from weight decay, following standard GPT-2/nanoGPT practice.
    All 2-D+ parameters — including the tied token embedding/output matrix —
    are penalized with ``config.WEIGHT_DECAY``.

    Args:
        model (torch.nn.Module): Model to collect parameters from.

    Returns:
        list[dict]: Two parameter-group dicts suitable for ``torch.optim.AdamW``.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or "bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": config.WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _step_optimizer(
    optimizer, scaler, all_params, opt_step, total_steps, warmup, max_lr
):
    """Apply gradient clipping and one optimizer step, then zero gradients.

    Handles both the GradScaler (fp16) and scaler-free (bf16/CPU) paths:
    gradients are unscaled before clipping only when a scaler is active.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to step.
        scaler (torch.cuda.amp.GradScaler | None): Active GradScaler, or
            ``None`` if not using fp16 loss scaling.
        all_params (list[torch.nn.Parameter]): All model parameters, used
            for gradient norm clipping.
        opt_step (int): Optimizer step counter *before* this step (will be
            incremented and returned).
        total_steps (int): Total steps for this stage's LR schedule.
        warmup (int): Warmup steps for this stage's LR schedule.
        max_lr (float): Peak LR for this stage's LR schedule.

    Returns:
        tuple[int, float]: ``(new_opt_step, lr_used)``.
    """
    opt_step += 1
    lr = cosine_lr(opt_step, total_steps, warmup, max_lr, config.MIN_LR)
    for group in optimizer.param_groups:
        group["lr"] = lr

    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(all_params, config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(all_params, config.GRAD_CLIP)
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    return opt_step, lr


@torch.no_grad()
def evaluate(model, val_loader, pad_id):
    """Compute mean cross-entropy loss over a validation loader.

    Label smoothing is deliberately disabled here (unlike the training loss)
    so the reported metric is a calibration-comparable measure to track
    across evaluations, uninflated by the smoothing term used to regularise
    training. Padding/orphan-fragment positions are excluded via
    ``ignore_index=pad_id``, matching the training loss's masking.

    Args:
        model (torch.nn.Module): Model to evaluate (switched to eval mode
            internally and restored to train mode before returning).
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        pad_id (int): Token ID to ignore in the loss (padding / orphan
            fragments).

    Returns:
        float: Mean per-token cross-entropy loss over the validation set.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    autocast_dtype, _ = _make_amp_config()
    for x, y in val_loader:
        x, y = (
            x.to(config.DEVICE, non_blocking=True),
            y.to(config.DEVICE, non_blocking=True),
        )
        ctx = (
            torch.autocast(device_type=config.DEVICE.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else torch.no_grad()
        )
        with ctx:
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=pad_id,
                reduction="sum",
            )
        n_valid = (y != pad_id).sum().item()
        total_loss += loss.item()
        total_tokens += max(1, n_valid)
    model.train()
    return total_loss / max(1, total_tokens)


def train_model(
    raw_model,
    model,
    train_loader,
    val_loader,
    epochs,
    max_lr,
    warmup,
    pad_id,
    best_ckpt_path,
    patience,
):
    """Run the full train/validate/early-stop loop for one curriculum stage.

    Combines gradient accumulation (to reach ``config.LOGICAL_BATCH`` from
    ``config.PHYSICAL_BATCH``-sized micro-batches), mixed precision, cosine
    LR scheduling, gradient clipping, and validation-based early stopping
    with best-checkpoint persistence. The checkpoint at *best_ckpt_path* is
    only overwritten when validation loss improves by more than ``1e-4``.
    At the end of the stage, if a best checkpoint was saved, it is reloaded
    into *raw_model* so the caller always ends up with the best-validation
    weights rather than the final epoch's weights.

    Args:
        raw_model (torch.nn.Module): The uncompiled model (used for
            checkpoint state_dict save/load).
        model (torch.nn.Module): The (possibly ``torch.compile``-wrapped)
            model used for the forward pass.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader | None): Validation loader,
            or ``None`` if no validation split is available for this stage.
        epochs (int): Maximum number of epochs (a ceiling — early stopping
            may end training sooner).
        max_lr (float): Peak learning rate for this stage.
        warmup (int): Warmup steps for this stage.
        pad_id (int): Token ID to ignore in the loss.
        best_ckpt_path (str): Path to persist the best-validation checkpoint.
        patience (int): Number of consecutive non-improving evaluations
            before early stopping triggers.
    """
    has_val = val_loader is not None
    autocast_dtype, scaler = _make_amp_config()
    all_params = list(raw_model.parameters())
    optimizer = torch.optim.AdamW(
        _make_param_groups(raw_model), lr=max_lr, betas=(0.9, 0.95)
    )

    total_steps = epochs * math.ceil(len(train_loader) / config.ACCUM_STEPS)

    best_val = float("inf")
    no_improve = 0
    opt_step = 0
    lr = 0.0
    stop = False
    last_eval_step = -1

    def check_val(tag):
        """Evaluate on the validation set, save the checkpoint if improved, and check early stopping."""
        nonlocal best_val, no_improve, stop, last_eval_step
        last_eval_step = opt_step
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
        micro_step = 0
        running, running_n = 0.0, 0

        for x, y in train_loader:
            x, y = (
                x.to(config.DEVICE, non_blocking=True),
                y.to(config.DEVICE, non_blocking=True),
            )
            ctx = (
                torch.autocast(device_type=config.DEVICE.type, dtype=autocast_dtype)
                if autocast_dtype is not None
                else torch.enable_grad()
            )
            with ctx:
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=pad_id
                )
            scaled_loss = loss / config.ACCUM_STEPS
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            running += loss.item()
            running_n += 1
            micro_step += 1

            if micro_step % config.ACCUM_STEPS == 0:
                opt_step, lr = _step_optimizer(
                    optimizer, scaler, all_params, opt_step, total_steps, warmup, max_lr
                )

                if opt_step % config.LOG_INTERVAL == 0:
                    log(
                        f"Epoch {epoch}/{epochs}  step {opt_step}/{total_steps}"
                        f"  lr={lr:.2e}  loss={running / max(1, running_n):.4f}"
                    )
                    running, running_n = 0.0, 0

                if has_val and opt_step % config.EVAL_INTERVAL_STEPS == 0:
                    check_val(f"Epoch {epoch}/{epochs}  step {opt_step}/{total_steps}")
                    if stop:
                        break

        if stop:
            break

        if micro_step % config.ACCUM_STEPS != 0:
            opt_step, lr = _step_optimizer(
                optimizer, scaler, all_params, opt_step, total_steps, warmup, max_lr
            )
            if running_n > 0:
                log(
                    f"Epoch {epoch}/{epochs}  step {opt_step}/{total_steps}"
                    f"  lr={lr:.2e}  loss={running / max(1, running_n):.4f}"
                )

        if has_val and opt_step != last_eval_step:
            check_val(f"Epoch {epoch}/{epochs}")
        elif not has_val:
            log(f"Epoch {epoch}/{epochs} complete")

        if stop:
            break

    if has_val and os.path.exists(best_ckpt_path):
        raw_model.load_state_dict(
            torch.load(best_ckpt_path, map_location=config.DEVICE, weights_only=True)
        )
        log(f"Restored best checkpoint (val_loss={best_val:.4f})")


def _log_stage(name, epochs, lr, train_games, val_games):
    """Log a stage-start banner with its key hyperparameters.

    Args:
        name (str): Stage name (e.g. ``"Stage 1 (pretraining)"``).
        epochs (int): Maximum epochs configured for this stage.
        lr (float): Peak learning rate for this stage.
        train_games (int): Number of training sequences/windows.
        val_games (int): Number of validation sequences/windows.
    """
    log(f"── {name} ──")
    log(
        f"  epochs(max)={epochs}  lr={lr:.1e}  train_windows={train_games:,}  val_windows={val_games:,}"
    )


def _run_stage(
    raw_model,
    vocab,
    train_data,
    val_data,
    epochs,
    lr,
    warmup,
    ckpt_path,
    stage_name,
    patience,
):
    """Train one curriculum stage, or skip it if its checkpoint already exists.

    Supports inter-stage resumption: if *ckpt_path* already exists on disk,
    the stage is assumed complete and its weights are loaded directly
    instead of retraining. This does not support resuming mid-stage — an
    interruption partway through a stage requires that stage to restart
    from scratch, but a fully completed stage is never repeated.

    Args:
        raw_model (torch.nn.Module): Model to train in place (or load
            weights into, if the stage is being skipped).
        vocab (dict[str, int]): Token-to-index mapping (used for pad/bos IDs).
        train_data (torch.Tensor): Training token stream for this stage.
        val_data (torch.Tensor): Validation token stream for this stage.
        epochs (int): Maximum epochs (ceiling) for this stage.
        lr (float): Peak learning rate for this stage.
        warmup (int): Warmup steps for this stage.
        ckpt_path (str): Path to this stage's best-checkpoint file, used
            both as the resume marker and the save target.
        stage_name (str): Human-readable stage name for logging.
        patience (int): Early-stopping patience for this stage.
    """
    if os.path.exists(ckpt_path):
        log(
            f"{stage_name}: checkpoint already exists at {ckpt_path} — skipping training"
        )
        raw_model.load_state_dict(
            torch.load(ckpt_path, map_location=config.DEVICE, weights_only=True)
        )
        return

    pad_id = vocab[config.PAD_TOK]
    bos_id = vocab[config.BOS_TOK]
    train_loader = make_loader(train_data, config.PHYSICAL_BATCH, True, bos_id, pad_id)
    val_loader = make_loader(val_data, config.PHYSICAL_BATCH, False, bos_id, pad_id)

    if train_loader is None:
        log(f"{stage_name}: no training data available — skipping")
        return

    _log_stage(
        stage_name,
        epochs,
        lr,
        len(train_loader.dataset),
        len(val_loader.dataset) if val_loader is not None else 0,
    )

    model = torch.compile(raw_model)
    model.train()
    train_model(
        raw_model,
        model,
        train_loader,
        val_loader,
        epochs,
        lr,
        warmup,
        pad_id,
        ckpt_path,
        patience,
    )


def run_train():
    """Load or build the training data and run the full two-stage curriculum.

    Single-responsibility entry point for the one-off training flow. Refuses
    to run if a final trained model is already present — training is meant to
    be a once-and-done operation; remove the artefact first if a full retrain
    is really wanted. Only ``config.MODEL_PATH`` is checked (not
    ``config.VOCAB_PATH``), since the vocab file is also written as an
    intermediate artefact partway through a normal run — guarding on it too
    would block resuming an interrupted run. Intermediate per-stage
    checkpoints are still reused to allow resuming an interrupted run.
    """
    if os.path.exists(config.MODEL_PATH):
        log(f"ERROR: {config.MODEL_PATH} already exists.")
        log(
            "Training is a one-off operation. Remove this file first if you really want to retrain."
        )
        return

    config.set_seed(config.SEED)
    vocab, s1_train, s1_val, s2_train, s2_val = load_or_build_data()

    raw_model = new_model(len(vocab))
    params = sum(p.numel() for p in raw_model.parameters())
    log(f"Model: {params:,} parameters, vocab={len(vocab):,}")

    _run_stage(
        raw_model,
        vocab,
        s1_train,
        s1_val,
        config.S1_EPOCHS,
        config.S1_LR,
        config.S1_WARMUP,
        config.S1_MODEL_PATH,
        "Stage 1 (pretraining)",
        config.EARLY_STOP_PATIENCE,
    )
    _run_stage(
        raw_model,
        vocab,
        s2_train,
        s2_val,
        config.S2_EPOCHS,
        config.S2_LR,
        config.S2_WARMUP,
        config.S2_MODEL_PATH,
        "Stage 2 (fine-tuning)",
        config.EARLY_STOP_PATIENCE,
    )

    tmp = config.MODEL_PATH + ".tmp"
    torch.save(raw_model.state_dict(), tmp)
    os.replace(tmp, config.MODEL_PATH)
    log(f"Training complete — saved {config.MODEL_PATH}")
