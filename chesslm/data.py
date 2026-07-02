import os
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from . import config
from .util import log
from .vocab import build_token_stream, build_vocab, load_vocab, save_vocab


def download_games(min_elo):
    """Stream the Hugging Face dataset and collect games above a minimum ELO threshold.

    Iterates over the full streaming split in a single pass, discarding samples
    where either player's ELO is missing or below *min_elo*, and where the game
    is shorter than 10 half-moves. Move annotations (check '+' and mate '#')
    are stripped from each SAN token before storing.

    Args:
        min_elo (int): Minimum ELO for both players. Games where either player
            is below this value are skipped.

    Returns:
        list[tuple[int, list[str]]]: A list of ``(elo, moves)`` pairs where
            *elo* is the lower of the two players' ratings and *moves* is the
            list of cleaned SAN move strings.
    """
    log(f"Streaming dataset: {config.DATASET_ID} (ELO >= {min_elo}) ...")
    ds = load_dataset(config.DATASET_ID, split="train", streaming=True)
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
    """Split a list of games into train and validation subsets.

    The split is reproducible via *seed*. If fewer than 20 games are provided
    the entire collection is returned as the training set and the validation
    set is empty, avoiding degenerate splits on tiny datasets.

    Args:
        games (list): Game records (any element type) to split.
        val_fraction (float): Fraction of games to place in the validation set,
            e.g. ``0.05`` for 5 %.
        seed (int): Random seed used to shuffle indices before splitting.

    Returns:
        tuple[list, list]: ``(train_games, val_games)``
    """
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


def _atomic_tensor_save(tensor, path):
    """Save a PyTorch tensor to *path* atomically via a temporary file.

    Writes to ``path + ".tmp"`` first and then uses :func:`os.replace` for an
    atomic rename, so a partially-written file is never visible at *path*.

    Args:
        tensor (torch.Tensor): Tensor to persist.
        path (str): Destination file path.
    """
    tmp = path + ".tmp"
    torch.save(tensor, tmp)
    os.replace(tmp, path)


class ChessDataset(Dataset):
    """PyTorch ``Dataset`` over a flat token-ID stream for next-token prediction.

    The stream is split into non-overlapping context windows of length
    ``config.CONTEXT_LEN``. Each item returns an ``(x, y)`` pair where *x* is
    the input window and *y* is the same window shifted one position to the
    right, suitable for language-model training with cross-entropy loss.

    Args:
        data (torch.Tensor): 1-D tensor of token IDs (the full concatenated
            token stream).
        ctx (int): Context length (sequence length for each sample).
    """

    def __init__(self, data, ctx):
        n = (len(data) - 1) // ctx
        self.data = data[: n * ctx + 1]
        self.ctx = ctx
        self.n = n

    def __len__(self):
        """Return the number of non-overlapping context windows in the stream."""
        return self.n

    def __getitem__(self, idx):
        """Return the input/target pair for window *idx*.

        Args:
            idx (int): Window index in ``[0, len(self))``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(x, y)`` each of shape
                ``(ctx,)``, where ``y[i] == x[i + 1]``.
        """
        s = idx * self.ctx
        chunk = self.data[s : s + self.ctx + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


def make_loader(data, batch_size, train):
    """Wrap a token-ID tensor in a :class:`ChessDataset` and return a DataLoader.

    Returns ``None`` if the resulting dataset would be empty (e.g. when the
    token stream is shorter than one context window).

    Args:
        data (torch.Tensor): 1-D token-ID tensor.
        batch_size (int): Number of sequences per batch.
        train (bool): Whether to shuffle and drop the last incomplete batch.
            When ``False`` (validation) the data is iterated in order without
            dropping samples.

    Returns:
        torch.utils.data.DataLoader | None: Configured loader, or ``None`` if
            the dataset is empty.
    """
    dataset = ChessDataset(data, config.CONTEXT_LEN)
    if len(dataset) == 0:
        return None
    n_workers = min(4, os.cpu_count() or 1)
    g = torch.Generator()
    g.manual_seed(config.SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=n_workers,
        pin_memory=(config.DEVICE.type == "cuda"),
        persistent_workers=(n_workers > 0),
        drop_last=train,
        generator=g if train else None,
    )


def load_or_build_data():
    """Load pre-processed data from disk or build it from scratch.

    Checks whether all five cache files (vocab, stage-1 train/val tensors,
    stage-2 train/val tensors) already exist. If they do, they are loaded
    directly; otherwise the Hugging Face dataset is streamed, the vocabulary
    is built, games are tokenised, and all artefacts are written atomically to
    the paths defined in :mod:`config`.

    The dataset download is performed in a *single* streaming pass: games are
    partitioned into the two ELO bands (stage 1 and stage 2) during the same
    pass that collects them.

    Returns:
        tuple: ``(vocab, s1_train, s1_val, s2_train, s2_val)`` where *vocab*
            is the token-to-index mapping (dict) and the remaining four
            elements are 1-D :class:`torch.Tensor` objects containing token
            IDs for each split.
    """
    caches = [
        config.VOCAB_PATH,
        config.S1_TRAIN_CACHE,
        config.S1_VAL_CACHE,
        config.S2_TRAIN_CACHE,
        config.S2_VAL_CACHE,
    ]
    if all(os.path.exists(p) for p in caches):
        log("Data cache found — loading ...")
        vocab = load_vocab(config.VOCAB_PATH)
        return (
            vocab,
            torch.load(config.S1_TRAIN_CACHE, weights_only=True),
            torch.load(config.S1_VAL_CACHE, weights_only=True),
            torch.load(config.S2_TRAIN_CACHE, weights_only=True),
            torch.load(config.S2_VAL_CACHE, weights_only=True),
        )

    log("Downloading data (single pass over the dataset) ...")
    pool = download_games(config.S1_MIN_ELO)
    s1_pool = [moves for elo, moves in pool if elo < config.S2_MIN_ELO]
    s2_pool = [moves for elo, moves in pool if elo >= config.S2_MIN_ELO]
    log(
        f"Stage 1 pool [{config.S1_MIN_ELO}, {config.S2_MIN_ELO}): {len(s1_pool):,} games"
    )
    log(f"Stage 2 pool >= {config.S2_MIN_ELO}: {len(s2_pool):,} games")

    s1_train_g, s1_val_g = split_games(s1_pool, config.VAL_FRACTION, config.SEED)
    s2_train_g, s2_val_g = split_games(s2_pool, config.VAL_FRACTION, config.SEED + 1)
    log(f"Stage 1 split — train: {len(s1_train_g):,}  val: {len(s1_val_g):,}")
    log(f"Stage 2 split — train: {len(s2_train_g):,}  val: {len(s2_val_g):,}")

    vocab = build_vocab(s1_train_g + s1_val_g + s2_train_g + s2_val_g)
    save_vocab(vocab, config.VOCAB_PATH)
    log(f"Vocab: {len(vocab)} tokens")

    s1_train = build_token_stream(s1_train_g, vocab)
    _atomic_tensor_save(s1_train, config.S1_TRAIN_CACHE)
    s1_val = build_token_stream(s1_val_g, vocab)
    _atomic_tensor_save(s1_val, config.S1_VAL_CACHE)
    s2_train = build_token_stream(s2_train_g, vocab)
    _atomic_tensor_save(s2_train, config.S2_TRAIN_CACHE)
    s2_val = build_token_stream(s2_val_g, vocab)
    _atomic_tensor_save(s2_val, config.S2_VAL_CACHE)
    log("Data cache saved")

    return vocab, s1_train, s1_val, s2_train, s2_val
