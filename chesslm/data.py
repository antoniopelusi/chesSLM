from collections import namedtuple

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from .config import BOS, DATASET_ID, DEVICE, PAD, SPECIAL_TOKENS, cfg

CTX = cfg.model.context_len
BUCKET = 16

assert CTX % BUCKET == 0

Pool = namedtuple("Pool", "tokens starts lens")


def load_data():
    file = pq.ParquetFile(
        hf_hub_download(DATASET_ID, "train.parquet", repo_type="dataset")
    )
    vocab = dict(SPECIAL_TOKENS)
    tokens, lens, elo = [], [], []
    for i in range(file.num_row_groups):
        group = file.read_row_group(i)
        games = group.column("moves").combine_chunks()
        flat = games.flatten()
        ids = np.array(
            [vocab.setdefault(m, len(vocab)) for m in flat.dictionary.to_pylist()],
            dtype=np.uint16,
        )
        size = np.diff(games.offsets.to_numpy())
        moves = ids[flat.indices.to_numpy()]
        tokens.append(np.insert(moves, np.cumsum(size) - size, BOS))
        lens.append(size + 1)
        white = group.column("white_elo").fill_null(0).to_numpy()
        black = group.column("black_elo").fill_null(0).to_numpy()
        elo.append(np.minimum(white, black))

    tokens, lens, elo = map(np.concatenate, (tokens, lens, elo))
    assert lens.max() - 1 <= CTX, "context_len is shorter than the longest game"
    starts = np.cumsum(lens) - lens
    elite = elo >= cfg.data.stage2_min_elo
    val = (
        np.random.default_rng(cfg.train.seed).random(len(lens)) < cfg.data.val_fraction
    )

    def pool(mask):
        return Pool(tokens, starts[mask], lens[mask])

    pools = {
        "stage1": pool(~elite & ~val),
        "stage2": pool(elite & ~val),
        "stage1_val": pool(~elite & val),
        "stage2_val": pool(elite & val),
    }
    return vocab, pools


def plan(pool, rng):
    x_len = pool.lens - 1
    width = -(-x_len // BUCKET) * BUCKET
    chunks = []
    for w in np.unique(width):
        games = rng.permutation(np.flatnonzero(width == w))
        size = cfg.train.micro_tokens // w
        chunks += [(w, games[i : i + size]) for i in range(0, len(games), size)]
    return [chunks[i] for i in rng.permutation(len(chunks))]


def batches(pool, rng):
    for w, games in plan(pool, rng):
        cols = np.arange(w + 1)
        pos = pool.starts[games][:, None] + cols
        seq = pool.tokens[np.minimum(pos, len(pool.tokens) - 1)].astype(np.int64)
        seq[cols >= pool.lens[games][:, None]] = PAD
        seq = torch.from_numpy(seq)
        x, y = seq[:, :-1].contiguous(), seq[:, 1:].contiguous()
        yield x.to(DEVICE), y.to(DEVICE)
