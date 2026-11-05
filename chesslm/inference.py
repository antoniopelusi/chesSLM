import json
import random
import sys

import chess
import torch
import torch.nn.functional as F

from .config import BOS, DEVICE, PAD, cfg, set_seed
from .model import load_checkpoint
from .util import log


class Engine:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab
        self.tokens = []
        self.past = None

    def next_logits(self, game_moves):
        ids = [BOS] + [self.vocab.get(m, PAD) for m in game_moves]
        ids = ids[-self.model.arch["context_len"] :]
        keep = 0
        for a, b in zip(self.tokens, ids):
            if a != b:
                break
            keep += 1
        keep = min(keep, len(ids) - 1)
        x = torch.tensor([ids[keep:]], dtype=torch.long, device=DEVICE)
        autocast = torch.autocast(
            DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"
        )
        with torch.inference_mode(), autocast:
            past = (
                [(k[:, :, :keep], v[:, :, :keep]) for k, v in self.past]
                if keep
                else None
            )
            logits, self.past = self.model.step(x, past)
        self.tokens = ids
        return logits[0, -1].float()

    def best_move(self, board, game_moves):
        moves = list(board.legal_moves)
        logits = self.next_logits(game_moves)

        legal = []
        for move in moves:
            token_id = self.vocab.get(board.san(move).rstrip("+#"))
            if token_id is not None:
                legal.append((token_id, move))
        if not legal:
            return random.choice(moves).uci()

        scores = logits[torch.tensor([t for t, _ in legal], device=DEVICE)]
        temperature = cfg.inference.temperature
        if temperature <= 0.0:
            idx = scores.argmax().item()
        else:
            idx = torch.multinomial(F.softmax(scores / temperature, dim=0), 1).item()
        return legal[idx][1].uci()


def parse_position(parts):
    board = chess.Board()
    i = 1
    if parts[i : i + 1] == ["startpos"]:
        i += 1
    elif parts[i : i + 1] == ["fen"]:
        board = chess.Board(" ".join(parts[i + 1 : i + 7]))
        i += 7

    game_moves = []
    if parts[i : i + 1] == ["moves"]:
        for uci in parts[i + 1 :]:
            move = chess.Move.from_uci(uci)
            game_moves.append(board.san(move).rstrip("+#"))
            board.push(move)
    return board, game_moves


def set_option(parts):
    if "name" not in parts or "value" not in parts:
        return
    name_idx, value_idx = parts.index("name") + 1, parts.index("value")
    if " ".join(parts[name_idx:value_idx]).lower() != "temperature":
        return
    value = " ".join(parts[value_idx + 1 :])
    try:
        cfg.inference.temperature = float(value)
        log(f"Temperature: {cfg.inference.temperature}")
    except ValueError:
        log(f"ERROR: invalid temperature {value!r}")


def load_engine(model_path, vocab_path):
    with open(vocab_path) as f:
        vocab = json.load(f)
    log(f"Loading {model_path} ...")
    model = load_checkpoint(model_path)
    log(f"Vocab: {len(vocab):,} tokens (padded to {model.tok_emb.num_embeddings:,})")
    log(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    log("Ready")
    return Engine(model, vocab)


def uci_loop(model_path, vocab_path):
    board = chess.Board()
    game_moves = []
    engine = None

    for line in sys.stdin:
        cmd = line.strip()
        if not cmd:
            continue

        if cmd == "uci":
            print("id name chesSLM")
            print("id author Antonio Pelusi")
            default = cfg.inference.temperature
            print(f"option name Temperature type string default {default}")
            print("uciok", flush=True)
            engine = engine or load_engine(model_path, vocab_path)
        elif cmd.startswith("setoption"):
            set_option(cmd.split())
        elif cmd == "isready":
            print("readyok")
        elif cmd == "ucinewgame":
            board, game_moves = chess.Board(), []
        elif cmd.startswith("position"):
            board, game_moves = parse_position(cmd.split())
        elif cmd.startswith("go"):
            print(f"bestmove {engine.best_move(board, game_moves)}")
        elif cmd == "quit":
            break

        sys.stdout.flush()


def run_inference(model_path, vocab_path):
    set_seed(cfg.train.seed)
    uci_loop(model_path, vocab_path)
