import json
import os

import torch

from .config import BOS_TOK, EOS_TOK, PAD_TOK


def build_vocab(games):
    """Build a token-to-index vocabulary from a collection of games.

    Collects every unique SAN move string that appears in *games* and adds
    the three special tokens (PAD, BOS, EOS) defined in :mod:`config`. Tokens
    are sorted alphabetically before assigning indices so that the vocabulary
    is deterministic regardless of iteration order.

    Args:
        games (list[list[str]]): Each element is a list of SAN move strings
            for one game (check/mate annotations should already be stripped).

    Returns:
        dict[str, int]: Mapping from token string to integer index.
    """
    tokens = {PAD_TOK, BOS_TOK, EOS_TOK}
    for game in games:
        tokens.update(game)
    return {t: i for i, t in enumerate(sorted(tokens))}


def save_vocab(vocab, path):
    """Serialise *vocab* to a JSON file at *path* atomically.

    Writes to ``path + ".tmp"`` first, then uses :func:`os.replace` for an
    atomic rename so a partially-written file is never visible at *path*.

    Args:
        vocab (dict[str, int]): Token-to-index mapping to persist.
        path (str): Destination file path.
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(vocab, f)
    os.replace(tmp, path)


def load_vocab(path):
    """Load a vocabulary mapping from a JSON file.

    Args:
        path (str): Path to a JSON file previously written by
            :func:`save_vocab`.

    Returns:
        dict[str, int]: Token-to-index mapping.
    """
    with open(path) as f:
        return json.load(f)


def build_token_stream(games, vocab):
    """Convert a list of games into a flat 1-D tensor of token IDs.

    Each game is encoded as ``[BOS, move_1, move_2, ..., move_n, EOS]``.
    Move tokens that are not present in *vocab* are silently skipped (this
    should not happen if the vocab was built from the same game pool, but
    the guard prevents crashes on edge cases).

    All per-game sequences are concatenated into a single stream so that the
    resulting tensor can be sliced into fixed-length context windows by
    :class:`~chesslm.data.ChessDataset`.

    Args:
        games (list[list[str]]): Each element is a list of SAN move strings
            for one game.
        vocab (dict[str, int]): Token-to-index mapping returned by
            :func:`build_vocab` or :func:`load_vocab`.

    Returns:
        torch.Tensor: 1-D ``int64`` tensor of token IDs.
    """
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
