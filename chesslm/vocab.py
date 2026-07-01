import json
import os

import torch

from .config import BOS_TOK, EOS_TOK, PAD_TOK


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
