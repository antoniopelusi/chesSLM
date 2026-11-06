# ♟️ chesSLM
Chess **S**mall **L**anguage **M**odel

> Level: Stockfish ~1650 ELO at temperature 0.0

A decoder-only causal language model built from scratch in PyTorch to play chess. Games are tokenized as sequences of Standard Algebraic Notation (SAN) moves, one token per move, and the network predicts the next move. At inference a legal-move mask from `python-chess` guarantees rule-compliant play.

- Model: [antoniopelusi/chesSLM](https://huggingface.co/antoniopelusi/chesSLM)
- Dataset: [antoniopelusi/chesSLM-dataset](https://huggingface.co/datasets/antoniopelusi/chesSLM-dataset)

The dataset, the vocabulary and the model are three separate artifacts. The dataset holds only the cleaned, deduplicated games (SAN move lists with both ratings), so anyone can use it. Training tokenizes the games, builds the vocabulary and saves it as `chesslm_vocab.json`, and writes the model to `chesslm_model.pt`.

## Setup

```bash
make setup
```

`make setup` recreates the `.venv` virtual environment from scratch and installs `requirements.txt`. The CUDA environment is configured for a toolkit installed in `/usr/local/cuda`.

Get the weights, or train from scratch:

```bash
./chesSLM.py --getmodel
./chesSLM.py --train
```

`--getmodel` downloads `chesslm_model.pt` and `chesslm_vocab.json` from the model repository. `--train` trains from scratch with the settings in `config.json`, using the games of the dataset repository (downloaded once). It needs a CUDA GPU (about 73 GB of peak memory with the default settings) and refuses to run if `chesslm_model.pt` already exists.

## Play

chesSLM speaks UCI, so `./chesSLM.py` can be added as an engine to any UCI chess GUI (for example CuteChess, or Arena) to play against it or to run matches against other engines. Install CuteChess with `make cutechess`, then add `./chesSLM.py` as an engine with the repository root as working directory. It has been tested with CuteChess.

Playing needs only `./chesSLM.py`, with no argument: it looks for `chesslm_model.pt` and `chesslm_vocab.json` in the current working directory (the repository root). To play with a model stored elsewhere, pass the directory (not a file) that contains it, with the same two file names `chesslm_model.pt` and `chesslm_vocab.json`: `./chesSLM.py --model /path/to/directory` (in a GUI, put `--model /path/to/directory` in the engine arguments). The model must have been produced by this code.

The engine answers `uciok` immediately and loads the model afterwards, so `readyok` is sent only when the model is ready. The `Temperature` engine option controls sampling over legal moves. `0.0` is greedy argmax; values between 0.0 and 0.3 are recommended. It runs on the GPU if one is available, otherwise on the CPU.

## How it works

One entry point and one package. This section explains the data, the model, the training and the results, then walks through the three flows.

### Data

The dataset has one row per game, with the moves in SAN (without check and mate marks) and the ratings of White and Black. It is the union of several public collections, cleaned once and uploaded as a single file: 14,100,262 complete games, 5,498 distinct moves, 1,180,715,061 plies, between 11 and 692 plies per game. Exact duplicates were removed (first occurrence kept). Source composition and columns are in the dataset card. There is no game result, only the ratings.

At the start of training the code turns the games into tokens:

- **Vocabulary.** `<pad>` is 0 and `<bos>` is 1; every distinct move then gets the next id in order of first appearance. The result is deterministic: the same dataset always yields the same vocabulary (5,500 tokens). There is no end token.
- **One game per row.** A training sequence is `<bos> m1 ... mn`, exactly what the model sees when playing. There is no packing of several games into one sequence, no cross-game attention and no cut fragments. All tokens live in one flat `uint16` array, with the start and length of each game.
- **Routing into two stages.** The Elo of a game is the lower of the two ratings (an unknown rating counts as 0). Games with Elo of at least `stage2_min_elo` (2500) go to stage 2, all others to stage 1. A random `val_fraction` (0.2%) of the games, drawn with a fixed seed, is held out as validation. The split is made by the code, not stored in the dataset. The result:

  | | Training games | Training tokens | Validation games |
  |---|---|---|---|
  | Stage 1 | 9,361,240 | 750M | 18,845 |
  | Stage 2 | 4,710,883 | 428M | 9,294 |

- **Batching.** Games are grouped by length into buckets of 16 tokens and each bucket is cut into micro-batches of `micro_tokens // width` games (at most `micro_tokens` padded tokens). The order of games and of micro-batches is reshuffled every epoch. Each game is right-padded with `<pad>` to the bucket width (padding stays under 10%). The input is the sequence without its last token and the target is the sequence without its first token, so every position predicts the next move; padding is ignored by the loss.

### Model

A GPT-style decoder, with the choices that are standard for this size:

| Component | Value |
|---|---|
| Parameters | 101,307,136 |
| Layers | 10 |
| Hidden size | 896 |
| Attention heads | 14 (head size 64) |
| Feed-forward | GELU, 4 × hidden size |
| Normalization | Pre-LayerNorm |
| Positions | Rotary (RoPE), base 10000 |
| Embeddings | tied input and output |
| Context length | 704 |
| Vocabulary | one token per move, plus `<pad>` and `<bos>`, padded to a multiple of 64 (5,504) |

Each block is `x + Attention(LN(x))` followed by `x + FFN(LN(x))`, where the feed-forward is two linear layers with a GELU in between; after the last block come a final LayerNorm and a projection onto the embedding matrix, which gives the logits over the vocabulary. For play, each layer also has a step mode that processes only the new tokens, reusing the cached keys and values of the previous ones. Attention uses a fused query/key/value projection, rotates queries and keys with RoPE (tables precomputed up to the context length) and applies causal scaled dot-product attention. There are no biases in the linear layers and no dropout. Initialization is GPT-2 style: standard deviation 0.02, with the residual projections scaled by `1/sqrt(2 * layers)`. The parameter count is `N = V*d + L*(12*d^2 + 4*d) + 2*d` with `V` the padded vocabulary size. The context length (704) is above the longest game (692 plies), so no game is ever cut.


### Training

- **Two stages, one model.** The same weights are trained first on stage 1 (all games below the Elo threshold), then on stage 2 (games between players rated 2500 or more). Stage 1 learns the rules and general play; stage 2 moves the model towards the moves of strong players. Each stage builds a fresh AdamW optimizer and its own schedule, and its step counter restarts from zero.
- **Optimizer.** AdamW with betas (0.9, 0.95), weight decay 0.1 on matrices only (the embedding included, the LayerNorm parameters excluded), gradient clipping at 1.0. Learning rate 4e-4 in stage 1 and 6e-5 in stage 2, 4 epochs each.
- **Schedule.** The learning rate is set at every step: linear warmup over `warmup_ratio` of the stage's steps (3% and 2%), then a cosine decay to 0.1 × the peak. There are 4,587 steps per epoch in stage 1 (18,348 in total) and 2,598 in stage 2 (10,392); the few micro-batches left over at the end of an epoch are dropped.
- **One optimizer step.** `batch_tokens` / `micro_tokens` = 1 micro-batch of up to 180,000 padded tokens per step (no gradient accumulation with the default settings). The loss is the sum of the cross-entropy (computed on float32 logits) over all real target tokens, divided by the number of real target tokens of the whole step, so every token weighs the same whatever the length of its game.
- **Precision and speed.** bfloat16 autocast with fp32 master weights, and the model is compiled with `torch.compile`.
- **Monitoring.** Every line of the log starts with the time. Training lines look like `stage1 ep 1/4 step 100/18348 lr 7.27e-05 mem 73.3 GB train_loss 6.3773 grad 3.52`: `mem` is the peak GPU memory, `train_loss` and `grad` are the averages of the loss and of the gradient norm before clipping since the previous report (at most 100 steps; the window restarts at the beginning of each epoch). Four times per epoch (the last one at the end of the epoch) a line reports `eval_train_loss` (a fixed random subset of the training games, the same size as the validation set) and `eval_val_loss` (the held-out games), computed without gradients on the same weights. A validation loss that stops falling or rises while the training loss keeps falling marks the onset of overfitting. The complete log of the training run of the published model is available in this repository as `training.log`.
- **No early stopping.** The final weights are those of the last step of stage 2.

### Results

| Metric | Value |
|---|---|
| Stockfish-equivalent level (temperature 0.0) | ~1650 ELO |
| Validation loss, end of stage 1 (games below 2500) | 1.423 |
| Validation loss, end of stage 2 (games rated 2500+) | 1.331 |

The final loss corresponds to a perplexity of about 3.8 equally likely moves per position. The validation loss decreased throughout both stages, and at the end the gap between validation and training loss (measured on the subset) is 0.06 in stage 1 and 0.04 in stage 2: the model generalizes and the last weights are also the best. The validation loss measures how well the model predicts held-out games, not how well it plays, which is why the level is measured by playing against Stockfish.

### The three flows

`chesSLM.py` is the only entry point. It first selects the GPU with the most free memory (and sets the CUDA environment), then reads the argument and checks that the chosen flow can run, so it fails immediately with a clear message instead of in the middle of the work. Progress messages go to standard error with a timestamp; standard output is reserved for the UCI protocol.

**`--train`**

1. Checks that `chesslm_model.pt` does not exist in the current directory and that CUDA is available.
2. Fixes the random seeds.
3. Downloads `train.parquet` from the dataset repository and reads it row group by row group: builds the vocabulary, inserts `<bos>`, computes the Elo of each game, checks that the longest game fits in the context length, and creates the four pools (stage 1, stage 2 and their validation sets).
4. Pads the vocabulary size to a multiple of 64, builds the model, prints the vocabulary size and the parameter count, and compiles it.
5. Runs stage 1, then stage 2 (the same function with different data and settings). It prints the size of the stage and its schedule, picks the training subset used for evaluation, and then for every step sets the learning rate, accumulates the gradients of the micro-batches, clips, updates and logs; four times per epoch it evaluates the training subset and the validation set.
6. Writes the weights and the architecture to `chesslm_model.pt` and the vocabulary to `chesslm_vocab.json`, in the current directory.

**`--getmodel`**

1. For each of `chesslm_model.pt` and `chesslm_vocab.json`: if the file already exists it is skipped.
2. Otherwise it is downloaded from the model repository and copied into the current directory.

**Play (no argument)**

1. Checks that the model and the vocabulary exist (in the current directory, or in the directory given with `--model`), otherwise stops suggesting `--getmodel`, `--train` or `--model DIR`. Then it fixes the seed and starts reading UCI commands from standard input.
2. On `uci` it answers with the engine name and author and the `Temperature` option, sends `uciok`, then loads the vocabulary and the model. The architecture is read from the checkpoint. Later commands wait until loading is done, so `readyok` is sent when the engine is ready.
3. On `position` it rebuilds the board (from `startpos` or a FEN, plus the moves played) and converts the moves to SAN without check and mate marks. Games that start from a FEN other than the initial position are outside what the model was trained on.
4. On `go` it builds the sequence `<bos>` followed by the game and finds how much of it is the same as the previous position, so that only the new tokens (always at least the last one) go through the model, with the cached keys and values of the shared prefix. Moves not in the vocabulary map to `<pad>`, and the sequence is cut to the last 704 tokens if it is longer.
5. It takes the logits of the last position, keeps only the legal moves that exist in the vocabulary, and answers with the best one (argmax when the temperature is 0, otherwise a sample from the softmax of the scores divided by the temperature). If no legal move is in the vocabulary it plays a random legal move.
6. `isready` answers `readyok`, `setoption name Temperature value x` changes the temperature, `ucinewgame` resets the game, `quit` ends the program. The reply to `go` ignores time-control parameters because the model answers immediately.

## Configuration

Every setting worth changing lives in `config.json`:

| Section | Keys |
|---|---|
| `model` | `d_model`, `n_layers`, `context_len` |
| `data` | `stage2_min_elo`, `val_fraction` |
| `train` | `seed`, `batch_tokens`, `micro_tokens`, `weight_decay` |
| `stage1`, `stage2` | `epochs`, `lr`, `warmup_ratio` |
| `inference` | `temperature` |

Fixed in code: head dimension 64 (so `n_heads = d_model / 64`), feed-forward width `4 * d_model`, RoPE base 10000, AdamW betas (0.9, 0.95), gradient clipping 1.0, final learning rate `0.1 * lr`, buckets of 16 tokens, 4 evaluations per epoch, a log line every 100 steps. `batch_tokens` is the number of (padded) tokens per optimizer step. `micro_tokens` is the number per forward pass and only affects memory and speed.

Checked when the program starts (the last one when training loads the data): `d_model` is a multiple of 64, `0 < val_fraction < 1`, `micro_tokens` divides `batch_tokens` and is at least `context_len`, `context_len` is a multiple of 16 and at least the longest game.

## Layout

- `chesSLM.py`: entry point (`--train`, `--getmodel`, `--model DIR`, UCI by default).
- `config.json`: all tunable settings.
- `training.log`: complete log of the training run of the published model.
- `chesslm/config.py`: loads `config.json`; constants, file names, device.
- `chesslm/model.py`: architecture and checkpoint save/load.
- `chesslm/data.py`: dataset loading, tokenization and length-bucketed batching.
- `chesslm/train.py`: two-stage training loop.
- `chesslm/getmodel.py`: pretrained model download.
- `chesslm/inference.py`: KV-cached engine, UCI loop, startup.
- `chesslm/util.py`: logging, CUDA environment and GPU selection.
- `Makefile`: `setup`, `reset_venv`, `cutechess`, `format`.
- `requirements.txt`: pinned dependencies.
