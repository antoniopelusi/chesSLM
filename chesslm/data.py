import os
import random

import chess
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from . import config
from .util import log
from .vocab import build_token_stream, build_vocab, load_vocab, save_vocab


def _known_elo(value):
    """Parse a raw ELO field into an int, or None if missing/unparsable.

    Args:
        value: Raw ``white_elo``/``black_elo`` field from a dataset sample —
            may be ``None``, a string, or a number.

    Returns:
        int | None: The parsed rating, or ``None`` if it can't be trusted.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def download_games():
    """Stream the full dataset once and split every usable game into stage 1 / stage 2.

    A game goes to stage 2 (fine-tuning) if at least one player has a known
    ELO >= ``config.S2_MIN_ELO``. Every other game — both ratings missing,
    only one present and below the threshold, or both present and below it —
    goes to stage 1 (pretraining). Stage 1 is therefore not ELO-filtered at
    all: a rating being missing or low is never itself a reason to discard a
    game, only a confirmed high rating routes it to fine-tuning.

    Games shorter than 10 half-moves are dropped outright. Each SAN move is
    additionally replayed against a :class:`chess.Board` purely to validate
    that it parses and is legal in sequence — a game is dropped as malformed
    if any move fails this check. The *stored* move representation remains
    the (check/mate-stripped) SAN string, not the UCI conversion produced
    incidentally by the validation step: this preserves the piece-identity/
    capture/check information SAN carries in the token itself, which plain
    coordinate (UCI) tokens do not.

    Returns:
        tuple[list[list[str]], list[list[str]]]: ``(stage1_games, stage2_games)``,
            each a list of cleaned-SAN move-string lists, one per game.
    """
    log(f"Streaming dataset: {config.DATASET_ID} ...")
    ds = load_dataset(config.DATASET_ID, split="train", streaming=True)
    s1_games, s2_games = [], []
    scanned = 0
    skipped_malformed = 0

    for sample in ds:
        scanned += 1
        san_moves = [m.rstrip("+#") for m in sample.get("moves_san", [])]
        if len(san_moves) < 10:
            continue

        board = chess.Board()
        malformed = False
        for san in san_moves:
            try:
                move = board.parse_san(san)
            except ValueError:
                malformed = True
                break
            board.push(move)

        if malformed:
            skipped_malformed += 1
            continue

        white_elo = _known_elo(sample.get("white_elo"))
        black_elo = _known_elo(sample.get("black_elo"))
        known = [e for e in (white_elo, black_elo) if e is not None]
        best_elo = max(known) if known else None

        if best_elo is not None and best_elo >= config.S2_MIN_ELO:
            s2_games.append(san_moves)
        else:
            s1_games.append(san_moves)

        if scanned % 100_000 == 0:
            log(
                f"  Scanned {scanned:,} — stage1 {len(s1_games):,}"
                f" — stage2 {len(s2_games):,}"
                f" — skipped {skipped_malformed:,} malformed ..."
            )

    log(
        f"Collected {len(s1_games):,} stage-1 and {len(s2_games):,} stage-2 games"
        f" from {scanned:,} samples ({skipped_malformed:,} discarded as malformed SAN)"
    )
    return s1_games, s2_games


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

    Games are packed back-to-back into one continuous stream (see
    :func:`build_token_stream` in :mod:`vocab`:
    ``[BOS, m1, ..., mk, EOS, BOS, m1, ..., EOS, ...]``) and sliced here into
    non-overlapping context windows of length ``config.CONTEXT_LEN``. Each
    window therefore typically spans several games, and a window that starts
    mid-game opens with a fragment whose preceding moves are outside the
    window — the true board state for that fragment is not reconstructable
    from what the model can see.

    To avoid training on that unreconstructable fragment, targets *before*
    the first BOS token in the window are replaced with ``pad_id`` and
    excluded from the loss via ``ignore_index`` (wired up in :mod:`train`).
    Once a BOS has been seen, every later position has its game's full move
    history inside the window, so its target is left untouched. If a window
    contains no BOS at all (only possible for games longer than
    ``config.CONTEXT_LEN`` plies), the whole window's targets are masked out,
    since no position in it has a reconstructable board state.

    Args:
        data (torch.Tensor): 1-D tensor of token IDs (the full concatenated
            token stream).
        ctx (int): Context length (sequence length for each sample).
        bos_id (int): Token ID of the BOS token, used to locate each
            window's first reconstructable position.
        pad_id (int): Token ID substituted into masked target positions;
            must match the ``ignore_index`` used for the training/validation
            loss.
    """

    def __init__(self, data, ctx, bos_id, pad_id):
        n = max(0, (len(data) - 1) // ctx)
        self.data = data[: n * ctx + 1]
        self.ctx = ctx
        self.bos_id = bos_id
        self.pad_id = pad_id
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
                ``(ctx,)``, where ``y[i] == x[i + 1]``, except that ``y[i]``
                is set to ``pad_id`` for every ``i`` before the window's
                first BOS token, or for the entire window if it contains no
                BOS at all.
        """
        s = idx * self.ctx
        chunk = self.data[s : s + self.ctx + 1]
        x, y = chunk[:-1].clone(), chunk[1:].clone()
        bos_positions = (x == self.bos_id).nonzero(as_tuple=True)[0]
        if len(bos_positions) > 0:
            first_bos = bos_positions[0].item()
            y[:first_bos] = self.pad_id
        else:
            y[:] = self.pad_id
        return x, y


def make_loader(data, batch_size, train, bos_id, pad_id):
    """Wrap a token-ID tensor in a :class:`ChessDataset` and return a DataLoader.

    Returns ``None`` if the resulting dataset would be empty (e.g. when the
    token stream is shorter than one context window).

    Args:
        data (torch.Tensor): 1-D token-ID tensor.
        batch_size (int): Number of sequences per batch.
        train (bool): Whether to shuffle and drop the last incomplete batch.
            When ``False`` (validation) the data is iterated in order without
            dropping samples.
        bos_id (int): BOS token ID, forwarded to :class:`ChessDataset`.
        pad_id (int): PAD token ID, forwarded to :class:`ChessDataset` and
            used to mask orphan-fragment targets out of the loss.

    Returns:
        torch.utils.data.DataLoader | None: Configured loader, or ``None`` if
            the dataset is empty.
    """
    dataset = ChessDataset(data, config.CONTEXT_LEN, bos_id, pad_id)
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
    directly; otherwise the Hugging Face dataset is streamed (with malformed-
    game validation, see :func:`download_games`), the SAN vocabulary is
    mined from the collected games, games are tokenised, and all artefacts
    are written atomically to the paths defined in :mod:`config`.

    The dataset download is a *single* streaming pass: each game is routed to
    stage 1 or stage 2 during the same pass that collects it (see
    :func:`download_games` for the routing rule).

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
    s1_pool, s2_pool = download_games()
    log(f"Stage 1 pool (no rating floor): {len(s1_pool):,} games")
    log(f"Stage 2 pool (best known ELO >= {config.S2_MIN_ELO}): {len(s2_pool):,} games")

    s1_train_g, s1_val_g = split_games(s1_pool, config.VAL_FRACTION, config.SEED)
    s2_train_g, s2_val_g = split_games(s2_pool, config.VAL_FRACTION, config.SEED + 1)
    log(f"Stage 1 split — train: {len(s1_train_g):,}  val: {len(s1_val_g):,}")
    log(f"Stage 2 split — train: {len(s2_train_g):,}  val: {len(s2_val_g):,}")

    vocab = build_vocab(s1_train_g + s1_val_g + s2_train_g + s2_val_g)
    save_vocab(vocab, config.VOCAB_PATH)
    log(f"Vocab: {len(vocab)} tokens (mined from SAN move strings)")

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
