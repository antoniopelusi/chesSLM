import random
import sys

import chess
import torch
import torch.nn.functional as F

from . import config
from .util import log


def get_best_move(model, vocab, board, game_moves):
    pad_id = vocab[config.PAD_TOK]
    ids = [vocab[config.BOS_TOK]] + [vocab.get(m, pad_id) for m in game_moves]
    ids = ids[-config.CONTEXT_LEN :]
    real_len = len(ids)
    ids = ids + [pad_id] * (config.CONTEXT_LEN - real_len)
    x = torch.tensor([ids], dtype=torch.long, device=config.DEVICE)

    with torch.inference_mode():
        logits = model(x)[0, real_len - 1]

    legal_map = {board.san(m).rstrip("+#"): m for m in board.legal_moves}
    legal_tokens = [
        (san, vocab[san], move) for san, move in legal_map.items() if san in vocab
    ]

    if not legal_tokens:
        return random.choice(list(board.legal_moves)).uci()

    tids = torch.tensor([tid for _, tid, _ in legal_tokens], device=config.DEVICE)
    scores = logits[tids]

    # Read TEMPERATURE through the module so runtime setoption changes are visible.
    if config.TEMPERATURE <= 0.0:
        idx = scores.argmax().item()
    else:
        probs = F.softmax(scores / config.TEMPERATURE, dim=0)
        idx = torch.multinomial(probs, 1).item()

    return legal_tokens[idx][2].uci()


def warmup_model(model, vocab):
    """Run a dummy inferences to trigger torch.compile JIT compilation."""
    board = chess.Board()
    get_best_move(model, vocab, board, [])
    if config.DEVICE.type == "cuda":
        torch.cuda.synchronize()


def uci_loop(model, vocab):
    board = chess.Board()
    game_moves = []

    for line in sys.stdin:
        cmd = line.strip()
        if not cmd:
            continue

        if cmd == "uci":
            print("id name chesSLM")
            print("id author Antonio Pelusi")
            print("option name Temperature type string default 0.0")
            print("uciok")

        elif cmd.startswith("setoption"):
            parts = cmd.split()
            if "name" in parts and "value" in parts:
                ni = parts.index("name") + 1
                vi = parts.index("value")
                name = " ".join(parts[ni:vi]).strip()
                value = " ".join(parts[vi + 1 :]).strip()
                if name.lower() == "temperature":
                    try:
                        config.TEMPERATURE = float(value)
                        log(f"Temperature set via setoption: {config.TEMPERATURE}")
                    except ValueError:
                        log(f"Invalid temperature value: {value!r}")

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
