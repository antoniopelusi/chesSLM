# ♟️ chesSLM
Chess **S**mall **L**anguage **M**odel

> Level: Stockfish ~XXX ELO at temperature 0.0

A decoder-only causal language model built from scratch in PyTorch to play chess. Games are tokenized as sequences of Standard Algebraic Notation (SAN) moves, one token per move, and the network predicts the next move. At inference a legal-move mask from `python-chess` guarantees rule-compliant play.

- Model: [antoniopelusi/chesSLM](https://huggingface.co/antoniopelusi/chesSLM)
- Dataset: [antoniopelusi/chesSLM-dataset](https://huggingface.co/datasets/antoniopelusi/chesSLM-dataset)

The dataset, the vocabulary and the model are three separate artifacts. The dataset holds only the cleaned, deduplicated games (SAN move lists with both ratings), so anyone can use it. Training tokenizes the games, builds the vocabulary and saves it as `chesslm_vocab.json`, and writes the model to `chesslm_model.pt`.

## Setup

```bash
make setup
```

Get the weights, or train from scratch:

```bash
./chesSLM.py --getmodel
./chesSLM.py --train
```

`--getmodel` downloads `chesslm_model.pt` and `chesslm_vocab.json` from the model repository. `--train` trains from scratch with the settings in `config.json`, using the games of the dataset repository (downloaded once). It refuses to run if `chesslm_model.pt` already exists.

## Play

chesSLM speaks UCI, so `./chesSLM.py` can be added as an engine to any UCI chess GUI (for example CuteChess, or Arena) to play against it or to run matches against other engines. Install CuteChess with `make cutechess`, then add `./chesSLM.py` as an engine with the repository root as working directory. It has been tested with CuteChess.

The engine answers `uciok` immediately and loads the model afterwards, so `readyok` is sent only when the model is ready. The `Temperature` engine option controls sampling over legal moves. `0.0` is greedy argmax; values between 0.0 and 0.3 are recommended.

## Results

| Metric | Value |
|---|---|
| Stockfish-equivalent level (temperature 0.0) | ~XXX ELO |
| Final validation loss | XXX |
| Training time | XXX hours on XXX |

## Model

| Component | Value |
|---|---|
| Parameters | 204,920,448 |
| Layers | 14 |
| Hidden size | 1088 |
| Attention heads | 17 (head size 64) |
| Feed-forward | GELU, 4 × hidden size |
| Normalization | Pre-LayerNorm |
| Positions | Rotary (RoPE), base 10000 |
| Embeddings | tied input and output |
| Context length | 704 |
| Vocabulary | one token per move, plus `<pad>` and `<bos>`, padded to a multiple of 64 |

No biases in the linear layers and no dropout. Initialization is GPT-2 style: standard deviation 0.02, residual projections scaled by `1/sqrt(2 * layers)`. The parameter count is `N = V*d + L*(12*d^2 + 4*d) + 2*d` with `V` the vocabulary size rounded up to a multiple of 64. Inference uses a KV cache that reuses the shared prefix between consecutive positions, so each move only processes the new tokens.

## Data

The dataset is built from several public game collections, with check and mate marks (`+`, `#`) removed from the moves and exact duplicate games removed (first occurrence kept), which leaves 14,100,262 games and 5,498 distinct moves. Nothing else is filtered: the source has no game under 11 plies and no malformed move (measured). The longest game has 692 plies, below the context length, so no game is ever cut. Source composition, columns and usage are in the dataset card.

## Training

1. **Tokenization.** At the start of training each distinct move gets an id, in order of first appearance after `<pad>` (0) and `<bos>` (1), and each game becomes `<bos> m1 ... mn`. Everything is deterministic: the same dataset always gives the same vocabulary.
2. **Routing.** Games where both ratings are known and at least `stage2_min_elo` go to stage 2; all others go to stage 1. A random `val_fraction` of the games (drawn with the seed `train.seed`, so it is the same in every run) is held out for validation.
3. **One game per row.** Every training row is a single game starting at `<bos>`, exactly like at inference: no packing across games, no cross-game attention, no masked fragments. Games are grouped by length into buckets of 16 tokens and batched with right padding (ignored by the loss) so that padding stays under 10%. Micro-batches are reshuffled every epoch and the loss is the exact mean over all real target tokens of the optimizer step.
4. **Stage 1 then stage 2.** Each stage runs a fixed number of epochs with linear warmup (`warmup_ratio` of the stage's total steps) and a full cosine decay to `0.1 * lr`. There is no early stopping: the final weights are the last step of stage 2. AdamW with decoupled weight decay on matrices only, betas (0.9, 0.95), gradient clipping 1.0, gradient accumulation over `batch_tokens / micro_tokens` micro-batches. Forward passes use bfloat16 autocast with fp32 master weights, and the model is compiled with `torch.compile`.
5. **Monitoring.** Four times per epoch the log reports `eval_train_loss` (a fixed random subset of the training games, the same size as the validation set) and `eval_val_loss` (the held-out games), measured on the same weights. A validation loss that stops falling or rises while the training loss keeps falling marks the onset of overfitting. Every 100 steps it also reports the running `train_loss` and the gradient norm before clipping. Both are monitors only and never used for early stopping.

## Configuration

Every setting worth changing lives in `config.json`:

| Section | Keys |
|---|---|
| `model` | `d_model`, `n_layers`, `context_len` |
| `data` | `stage2_min_elo`, `val_fraction` |
| `train` | `seed`, `batch_tokens`, `micro_tokens`, `weight_decay` |
| `stage1`, `stage2` | `epochs`, `lr`, `warmup_ratio` |
| `inference` | `temperature` |

Fixed in code: head dimension 64 (so `n_heads = d_model / 64`), feed-forward width `4 * d_model`, RoPE base 10000, AdamW betas (0.9, 0.95), gradient clipping 1.0, final learning rate `0.1 * lr`. `batch_tokens` is the number of (padded) tokens per optimizer step. `micro_tokens` is the number per forward pass and only affects memory and speed; it must divide `batch_tokens` and be at least `context_len`, and the number of accumulation steps is `batch_tokens / micro_tokens`.

## Layout

- `chesSLM.py`: entry point (`--train`, `--getmodel`, UCI by default).
- `config.json`: all tunable settings.
- `chesslm/config.py`: loads `config.json`; constants, file names, device.
- `chesslm/model.py`: architecture and checkpoint save/load.
- `chesslm/data.py`: dataset loading, tokenization and length-bucketed batching.
- `chesslm/train.py`: two-stage training loop.
- `chesslm/getmodel.py`: pretrained model download.
- `chesslm/inference.py`: KV-cached engine, UCI loop, startup.
- `chesslm/util.py`: logging, CUDA environment and GPU selection.
