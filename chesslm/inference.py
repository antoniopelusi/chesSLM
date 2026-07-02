import random
import sys

import chess
import torch
import torch.nn.functional as F

from . import config
from .util import log


def get_best_move(model, vocab, board, game_moves):
    """Select the best legal move for the current position using the language model.

    Builds a token-ID sequence from the game history, runs a forward pass, and
    scores only the legal moves available in *board*. The move with the highest
    score is returned when ``config.TEMPERATURE <= 0``; otherwise a move is
    sampled proportionally to the softmax probabilities scaled by the
    temperature. If none of the legal moves appear in the vocabulary, a random
    legal move is chosen as a fallback.

    The context is right-truncated to ``config.CONTEXT_LEN`` tokens and
    left-padded with the PAD token so that the tensor always has the expected
    fixed length.

    Args:
        model (torch.nn.Module): Compiled ChesSLM model in evaluation mode.
        vocab (dict[str, int]): Token-to-index mapping produced by
            :func:`~chesslm.vocab.build_vocab`.
        board (chess.Board): Current board state, used to enumerate legal moves
            and convert them to SAN notation.
        game_moves (list[str]): SAN move strings played so far in the game
            (without check/mate annotations), used to build the input context.

    Returns:
        str: UCI move string (e.g. ``"e2e4"``) for the selected move.
    """
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
    """Run a dummy inference to trigger ``torch.compile`` JIT compilation.

    The first forward pass through a compiled model is slow because PyTorch
    traces and compiles the computation graph. Calling this function before
    entering the UCI loop amortises that cost so the engine is ready to
    respond immediately when the GUI sends the first ``go`` command.

    On CUDA devices a :func:`torch.cuda.synchronize` call is issued after the
    dummy inference to ensure compilation has fully completed before returning.

    Args:
        model (torch.nn.Module): Compiled ChesSLM model in evaluation mode.
        vocab (dict[str, int]): Token-to-index mapping.
    """
    board = chess.Board()
    get_best_move(model, vocab, board, [])
    if config.DEVICE.type == "cuda":
        torch.cuda.synchronize()


def uci_loop(model, vocab):
    """Run the UCI (Universal Chess Interface) communication loop.

    Reads commands from ``stdin`` line by line and writes responses to
    ``stdout``, following the UCI protocol used by chess GUIs (e.g. Arena,
    Cutechess, Lichess Bot). The supported commands are:

    * ``uci`` — identify the engine and list options.
    * ``setoption name Temperature value <v>`` — update ``config.TEMPERATURE``
      at runtime without restarting the engine.
    * ``isready`` — respond ``readyok`` to signal the engine is initialised.
    * ``ucinewgame`` — reset the board and move history.
    * ``position [startpos | fen <fen>] [moves <uci_moves>]`` — set up the
      board from a starting position or FEN string and replay any moves.
    * ``go`` — compute and print the best move in ``bestmove <uci>`` format.
    * ``stop`` — acknowledged but no action is needed (search is synchronous).
    * ``quit`` — exit the loop.

    Args:
        model (torch.nn.Module): Compiled ChesSLM model in evaluation mode.
        vocab (dict[str, int]): Token-to-index mapping.
    """
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
