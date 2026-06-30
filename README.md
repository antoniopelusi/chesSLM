# ChesSLM — A GPT-Style Language Model That Plays Chess

> Level: `Stockfish ~1650 ELO` on temperature 0.0

ChesSLM is a single self-contained Python script that trains a small
**decoder-only Transformer** (architecturally a scaled-down GPT-2) to predict
chess moves, and then serves the trained model as a chess engine over the
**UCI protocol** (the standard text protocol used by chess GUIs such as
Arena, ChessBase, or `cutechess`/`lichess-bot`).

The core idea is deceptively simple: treat a chess game as a sequence of
*move tokens* and train a causal language model to predict "what move comes
next", exactly the way a text LLM predicts the next word. No board
representation, no hand-crafted features, no search tree — the model's only
job is sequence modeling. Legality is enforced afterwards, at inference
time, by intersecting the model's predictions with a real chess engine's
list of legal moves (`python-chess`).

This document explains **every implementation choice** in the script and
the **full technical pipeline**, end to end: data → tokenization → model →
training → inference → deployment.

---

## Table of contents

1. [High-level pipeline](#1-high-level-pipeline)
2. [Tokenization: moves as tokens](#2-tokenization-moves-as-tokens)
3. [Data pipeline](#3-data-pipeline)
4. [Model architecture](#4-model-architecture)
5. [Two-stage curriculum training](#5-two-stage-curriculum-training)
6. [Training mechanics](#6-training-mechanics)
7. [Checkpointing, caching and resumability](#7-checkpointing-caching-and-resumability)
8. [Inference and legality enforcement](#8-inference-and-legality-enforcement)
9. [Model warmup](#9-model-warmup)
10. [Serving the model: the UCI loop](#10-serving-the-model-the-uci-loop)
11. [Files produced by the pipeline](#11-files-produced-by-the-pipeline)
12. [Hyperparameter reference](#12-hyperparameter-reference)
13. [How to run it](#13-how-to-run-it)

---

## 1. High-level pipeline

The script is built around a single guiding principle: **idempotency**.
Every expensive artifact is cached to disk, and `main()` inspects which
artifacts already exist to decide what work is still needed. This means the
exact same command can be re-run after an interruption (crash, preemption,
manual `Ctrl-C`) and it will pick up exactly where it left off, never
redoing finished work.

```
                ┌───────────────────────────┐
                │ chesslm_model.pt exists?  │
                └──────────────┬────────────┘
                     yes ┌─────┴─────┐ no
                         │           │
                 Load + serve   Build/load data
                 over UCI       (download, filter,
                                 tokenize, cache)
                                     │
                          ┌──────────┴──────────┐
                          │ stage1 checkpoint?  │
                          └──┬───────────────┬──┘
                       exists│               │missing
                             │               │
                        skip │         Train Stage 1
                             │         (pretraining)
                             └──────┬────────┘
                                    │
                          ┌─────────┴───────────┐
                          │ stage2 checkpoint?  │
                          └──┬───────────────┬──┘
                       exists│               │missing
                             │               │
                        skip │         Train Stage 2
                             │         (finetuning)
                             └──────┬────────┘
                                    │
                          Save final model, warm up, serve over UCI
```

There are three independent "checkpoints" gating the pipeline:
the tokenized-data cache, the Stage-1 best checkpoint, and the Stage-2 best
checkpoint. Each is checked separately, so a run that completed Stage 1 but
was killed during Stage 2 resumes *exactly* at Stage 2 next time.

---

## 2. Tokenization: moves as tokens

Most chess neural networks operate on a *board* representation (e.g. an 8×8
tensor of piece planes) and output a *policy* over moves for that single
position. ChesSLM does something different and closer to how a text LLM
works: it never sees the board at all. It only ever sees a **flat sequence
of move tokens**, and learns to predict the next one.

Concretely, every distinct SAN (Standard Algebraic Notation) move string
that appears in the training data becomes one vocabulary token — e.g.
`"e4"`, `"Nf3"`, `"Qxd5"`, `"O-O"`, `"e8=Q"` are each a single, indivisible
token, exactly like a word in a text vocabulary. This is a **move-level**
tokenization scheme, as opposed to character-level (tokenizing the letters
"e", "4") or coordinate-level (tokenizing `"e2e4"` square pairs as in UCI
notation).

Two normalization choices keep this vocabulary clean and small:

* **Check/checkmate suffixes are stripped.** `"Qd5+"` and `"Qd5#"` both
  become `"Qd5"`. These suffixes are entirely determined by the resulting
  position and add nothing the model could not already infer, while
  needlessly multiplying the vocabulary size.
* **The vocabulary is built once, from the union of every split** (both
  curriculum stages, train and validation) **before any tokenization
  happens.** This guarantees Stage 1 and Stage 2 share exactly the same
  token IDs, which is essential since the same embedding table and output
  head are reused — and fine-tuned — across both stages.

Three special tokens complete the vocabulary: `<bos>` (beginning of a
game), `<eos>` (end of a game), and `<pad>` (reserved for padding).

Because the vocabulary is derived from the *finite* set of move strings
actually observed in the dataset (not the combinatorial space of every
syntactically possible SAN string), it stays a manageable, GPT-2-vocabulary-like
size — typically a few thousand tokens — small enough that a quadratic
attention mechanism and a tied embedding/output head are entirely
practical.

---

## 3. Data pipeline

### 3.1 Source and streaming

Games come from the Hugging Face dataset `angeluriot/chess_games`. The
dataset is **streamed** (`streaming=True`) rather than fully downloaded —
the script iterates over it sample-by-sample, discarding anything that does
not pass the filters below, which keeps memory usage and disk I/O bounded
regardless of how large the underlying dataset is.

### 3.2 ELO-based quality filtering

Each sample carries `white_elo` and `black_elo`. The script uses the
**minimum of the two** as the game's quality signal — a single weak player
is enough to introduce blunders and noise into an otherwise strong game, so
the *weaker* side's rating is the appropriate bottleneck to filter on, not
the average or the stronger side's rating.

A game is kept only if:
* both ratings are present and parse as integers,
* the weaker side's rating is **≥ 1800** (`S1_MIN_ELO`), and
* the game has **≥ 10 moves** (very short games — resignations,
  disconnections, etc. — are unrepresentative of normal play).

This single pass over the dataset (gated at the lowest threshold needed by
either stage) is then partitioned client-side into two pools:

| Pool | ELO range | Used for |
|---|---|---|
| Stage 1 pool | `[1800, 2400)` | Pretraining — broad coverage, larger volume |
| Stage 2 pool | `≥ 2400` | Finetuning — master-level games, smaller volume, higher quality |

Streaming the dataset once at the lowest threshold and splitting in memory
avoids the cost of two separate full passes over a potentially huge
dataset.

### 3.3 Train/validation split

Each pool is split into training and validation games with
`split_games()`. The split happens at the **game** level (not the
chunk/token level): an entire game is assigned wholly to train or wholly to
validation, which prevents validation chunks from being near-duplicates of
training chunks drawn from the same game. The split fraction is
`VAL_FRACTION = 0.03` (3%), with a fixed seed so it is reproducible; if a
pool has fewer than 20 games, validation is skipped entirely for that pool
(too little data to form a meaningful held-out set).

### 3.4 Token-stream packing

Once vocabulary and splits are fixed, each split is flattened into one long
1-D tensor of token IDs via `build_token_stream()`:

```
<bos> g1m1 g1m2 ... g1mK <eos> <bos> g2m1 g2m2 ... <eos> <bos> ...
```

All games in a split are concatenated back-to-back with **no padding**
between them. This "packing" strategy is the same trick used to pretrain
text LLMs on many documents: every position in every training example
carries a genuine supervision signal, with zero tokens wasted on padding.

The cost of packing is that a fixed-length training chunk can, in
principle, span the boundary between two unrelated games — the causal
attention mask is *not* reset at `<bos>`/`<eos>`. In practice the model
quickly learns that `<bos>` signals "ignore everything before this point",
since the previous game's continuation is never predictive of the new
game's opening.

### 3.5 Caching

Both the vocabulary (as JSON) and the four token-stream tensors (as
`torch.save`d files) are written **atomically**: each is first written to a
`*.tmp` path and then moved into place with `os.replace`, so a crash
mid-write never leaves a corrupted cache file that a later run could
mistakenly treat as valid.

---

## 4. Model architecture

ChesSLM (the `ChesSLM` class) is a standard **decoder-only, pre-LayerNorm
Transformer** — architecturally very close to GPT-2, with a few
modernizations.

```
      tokens ──► token embedding ──┐
                                   ├─ (+) ──► dropout ──► [Block] × N_LAYERS ──► LayerNorm ──► linear head ──► logits
positions ──► position embedding ──┘
```

Each `Block` is:

```
x ──► LayerNorm ──► CausalSelfAttention ──► (+ residual) ──► LayerNorm ──► FeedForward(GELU) ──► (+ residual)
```

### 4.1 Attention

`CausalSelfAttention` computes Q, K, V with a **single fused linear layer**
(one matmul instead of three separate projections) and delegates the actual
attention computation to
[`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
with `is_causal=True`. This lets PyTorch dispatch to a fused, memory-efficient
attention kernel (e.g. a FlashAttention-style implementation when
available) instead of materializing an explicit `T×T` masked score matrix —
faster and far less memory-hungry at the context lengths used here.

### 4.2 Pre-LayerNorm residual blocks

LayerNorm is applied **before** each sub-layer (pre-LN) rather than after
(post-LN, the original Transformer design). Pre-LN is the modern standard
because it keeps gradients well-behaved through deep stacks, allowing this
12-layer model to train without the careful learning-rate warmup
sensitivity that post-LN architectures require.

### 4.3 Feed-forward network

A standard 2-layer MLP with a GELU activation and the classic 4× expansion
ratio (`D_FF = 4096 = 4 × D_MODEL`), with dropout after each linear layer —
the same shape used in GPT-2/GPT-3.

### 4.4 Weight tying

The output head's weight matrix is **tied** to the token embedding
(`self.head.weight = self.tok_emb.weight`). This is a well-known trick
(used in GPT-2, AWD-LSTM, and many other LMs): the same matrix is used to
both *embed* a token and *predict* it, which roughly halves the parameter
count contributed by the (vocabulary × d_model) matrix and tends to improve
generalization, since the model is encouraged to keep semantically similar
tokens close together in the *same* space used for both reading and
writing.

### 4.5 Positional encoding

Positions are encoded with a **learned absolute positional embedding**
table (`nn.Embedding(ctx_len, d_model)`), exactly like GPT-2 — not rotary
embeddings (RoPE) or relative position biases (ALiBi). This is the simplest
choice and ties the model's maximum usable sequence length directly to
`CONTEXT_LEN`.

### 4.6 Initialization

Linear and Embedding weights are initialized from `N(0, 0.02²)` (the
GPT-2 initialization scheme); Linear biases start at zero. The tied `head`
module is explicitly **skipped** during this initialization pass, since
initializing it would simply overwrite the (already-initialized) shared
embedding weight a second time with a redundant random draw.

### 4.7 Size

| Hyperparameter | Value |
|---|---|
| Context length (`CONTEXT_LEN`) | 350 tokens |
| Model width (`D_MODEL`) | 1024 |
| Attention heads (`N_HEADS`) | 16 (head dim = 64) |
| Layers (`N_LAYERS`) | 12 |
| Feed-forward width (`D_FF`) | 4096 |
| Dropout | 0.15 |

With these settings the Transformer stack alone (attention + feed-forward +
LayerNorms, ×12 layers, plus the final LayerNorm) accounts for
**≈ 151.5M parameters**; add the positional embedding (`350 × 1024 ≈
0.36M`) and the (tied) token embedding/head (`vocab_size × 1024`, only
counted once thanks to tying), for a **total of roughly 150–155M
parameters** depending on the exact vocabulary size the dataset produces —
comparable in scale to GPT-2 Medium.

---

## 5. Two-stage curriculum training

Rather than training on one undifferentiated pool of games, ChesSLM uses a
**two-stage curriculum**, conceptually similar to "pretrain then
fine-tune" pipelines in modern LLM training — except here *both* stages
optimize the exact same self-supervised next-token objective; only the
*data distribution* and *learning-rate regime* change between stages.

| | Stage 1 — Pretraining | Stage 2 — Finetuning |
|---|---|---|
| Data | ELO ∈ `[1800, 2400)` — larger pool | ELO ≥ `2400` — smaller, master-level pool |
| Epochs | 8 | 10 |
| Peak LR | `4e-4` | `8e-5` (5× lower) |
| Warmup steps | 200 | 200 |
| Checkpoint | `chesslm_stage1_best.pt` | `chesslm_stage2_best.pt` |

**Rationale.** Stage 1 exposes the model to a large volume of solid (1800+)
but not necessarily elite play, building broad coverage of openings,
tactics, and typical move patterns. Stage 2 then continues training the
*same* weights on a smaller, higher-quality pool of 2400+ games with a much
smaller learning rate — gently nudging the model's move preferences toward
stronger play without the smaller, more specialized dataset causing
catastrophic forgetting of what was learned in Stage 1 (a low LR is the
standard mitigation for this in fine-tuning regimes). Both stages share the
exact same vocabulary and model architecture; Stage 2 simply continues
optimizing the weights Stage 1 produced.

---

## 6. Training mechanics

`train_model()` runs one full stage and bundles together the following
standard large-batch-Transformer training techniques:

### 6.1 Gradient accumulation

Optimizing with a very large batch size improves gradient estimate quality
but a batch of `LOGICAL_BATCH = 1024` sequences of 350 tokens does not fit
in GPU memory at once for a ~150M-parameter model. The script instead
processes `PHYSICAL_BATCH = 256` sequences per forward/backward pass and
accumulates gradients over `ACCUM_STEPS = LOGICAL_BATCH / PHYSICAL_BATCH =
4` micro-batches before taking a single optimizer step — simulating the
large logical batch size within a much smaller memory budget.

### 6.2 Mixed precision (AMP)

`_make_amp_config()` chooses the best numerical setup automatically:

* **CPU** → AMP disabled, plain float32.
* **CUDA GPU with bfloat16 support** (Ampere/Hopper and newer) → `bfloat16`
  autocast, **no** `GradScaler` needed (bf16's wide dynamic range makes
  manual loss scaling unnecessary).
* **CUDA GPU without bfloat16 support** (older architectures) → `float16`
  autocast **with** a `GradScaler`, to prevent gradient underflow.

### 6.3 Optimizer and regularization

* **AdamW** with `betas=(0.9, 0.95)` — a slightly faster-adapting second
  moment than the PyTorch default `(0.9, 0.999)`, following the recipe
  popularized by GPT-3-style training.
* **Decoupled weight decay = 0.15**, applied selectively via
  `_make_param_groups()`: only parameters with more than one dimension
  (i.e. weight matrices in `Linear` and `Embedding` layers) are decayed;
  1-D parameters (LayerNorm weights/biases) and any parameter with `"bias"`
  in its name are excluded from decay.
* **Gradient clipping** to a max norm of `1.0` (`GRAD_CLIP`) before every
  optimizer step, guarding against destabilizing gradient spikes.

### 6.4 Learning-rate schedule

`cosine_lr()` implements **linear warmup followed by cosine decay** down to
a floor of `MIN_LR = 1e-5` — the standard schedule for training
Transformer language models: the LR ramps up linearly over the first
`warmup` steps (avoiding instability from large updates while the Adam
moment estimates are still inaccurate), then follows a cosine curve down to
`min_lr` over the remaining steps.

### 6.5 Validation, checkpointing, and early stopping

If a stage has a non-empty validation split, the model is evaluated (mean
cross-entropy, dropout disabled, under `torch.inference_mode`) every
`EVAL_INTERVAL_STEPS = 150` optimizer steps *and* at the end of every
epoch. Whenever validation loss improves by more than `1e-4`, the current
weights are saved (atomically) as that stage's "best" checkpoint, and a
"no improvement" counter resets; otherwise the counter increments.
Training stops early once that counter reaches `EARLY_STOP_PATIENCE = 5`
consecutive non-improving evaluations.

After the stage's training loop ends (whether by exhausting all epochs or
by early stopping), if a validation set existed, the function **reloads the
best saved checkpoint** into the model before returning — so the stage
always hands off its *best*-validated weights to the next stage, not
whatever weights happened to be in memory when training stopped. If no
validation set was available for that pool, the final weights are simply
saved as-is (there is no validation signal to compare against).

### 6.6 `torch.compile`

The model is wrapped with `torch.compile()` for both training and
inference, letting PyTorch 2's JIT compiler fuse kernels and reduce Python
overhead. The original, uncompiled module (`raw_model`) is kept around
separately because it is what gets checkpointed (`state_dict()`) and what
the optimizer's parameter groups are built from — keeping these operations
decoupled from whatever internal wrapping `torch.compile` performs.

---

## 7. Checkpointing, caching and resumability

The pipeline writes — and checks for — five categories of artifacts, each
gating a different stage of work:

1. **Data caches** (vocabulary + 4 tokenized streams): if *all five* exist,
   `load_or_build_data()` skips the entire download/filter/tokenize
   pipeline.
2. **Stage 1 checkpoint**: if it exists, Stage 1 training is skipped and
   its weights are loaded directly.
3. **Stage 2 checkpoint**: same idea, for Stage 2.
4. **Final model**: if it (and the vocabulary) exists, `main()` skips
   *everything* — no data loading, no training — and goes straight to
   loading the model and serving it over UCI.

Every save in the pipeline (vocabulary, data tensors, stage checkpoints,
final model) uses the **write-to-temp-then-rename** pattern
(`os.replace`), which is atomic at the filesystem level: a process killed
mid-save can never leave behind a half-written file that a subsequent run
would mistake for a valid, complete artifact.

---

## 8. Inference and legality enforcement

`get_best_move()` is the single function bridging the language model and
an actual chess position. Its key design decision is to **never trust the
model's output as definitionally legal**:

1. The current game's move history is encoded the same way training data
   was: `<bos>` followed by each played move's vocabulary ID (an
   out-of-vocabulary move falls back to `<pad>`'s ID, a defensive measure
   for rare moves never seen at training time), truncated to the most
   recent `CONTEXT_LEN` tokens if the game has run longer than the model's
   context window.
2. One forward pass produces the next-token logits.
3. **`python-chess` computes the actual legal moves** for the current
   board and converts each to its check/mate-stripped SAN string. Only
   legal moves that also exist in the vocabulary are kept as candidates.
4. If, in some rare position, *no* legal move's SAN string is in the
   vocabulary, the function falls back to choosing uniformly at random
   among all legal moves — guaranteeing the engine can never crash or
   return an illegal move, no matter how unusual the position.
5. Otherwise, the model's logits are sliced down to just those legal
   candidates, and a move is chosen either by **greedy argmax**
   (`TEMPERATURE <= 0`, the default — always pick the legal move the model
   scores highest) or by **temperature-scaled sampling**
   (`TEMPERATURE > 0` — softmax the legal-move logits and sample, useful
   for generating varied, non-deterministic games).

In short: the neural network supplies a *ranking* over legal moves; a real
chess engine library supplies the *set* of legal moves; the intersection of
the two is what gets played.

---

## 9. Model warmup

`warmup_model()` runs three throwaway forward passes through `get_best_move()`
on a fresh starting position **before** the engine starts listening for UCI
commands. This exists purely to absorb a one-off cost that would otherwise
hit the *first real* `go` command: `torch.compile()` only traces and
compiles its fused kernels the first time the model is actually invoked
with a given input shape, and (on CUDA) that first invocation also pays for
context/kernel initialization. Without warmup, the GUI's very first move
request would stall for several seconds while this compilation happens;
running it eagerly at startup instead moves that cost to before the engine
announces it is ready.

On CUDA devices, `torch.cuda.synchronize()` is called after the warmup
passes to make sure the asynchronous GPU work has actually finished (and
not just been queued) before logging that warmup is complete.

Warmup runs in both startup paths: when a previously-trained model is
loaded straight from disk, and immediately after training finishes and the
model is about to be served for the first time — so the JIT-compilation
delay is absorbed exactly once, regardless of which path led to serving.

Note this is unrelated to the learning-rate *warmup* described in
[§6.4](#64-learning-rate-schedule) (`S1_WARMUP`/`S2_WARMUP`) — that warmup
shapes the optimizer's LR schedule during training; this warmup is a
one-time inference-side JIT/CUDA priming step that runs only at serving
startup, never during training.

---

## 10. Serving the model: the UCI loop

`uci_loop()` implements the subset of the
[UCI protocol](https://www.chessprogramming.org/UCI) needed to plug
ChesSLM into virtually any chess GUI or automated tournament tool:

| Command | Behavior |
|---|---|
| `uci` | Identify the engine, reply `uciok` |
| `isready` | Reply `readyok` |
| `ucinewgame` | Reset the internal board and move history |
| `position startpos\|fen ... [moves ...]` | Rebuild the board from scratch and replay the given moves, keeping the `chess.Board` and the SAN move-history list in sync |
| `go ...` | Run one forward pass via `get_best_move` and reply `bestmove <uci>` |
| `stop` | Accepted, no-op (there is no asynchronous search to interrupt — `go` is a single synchronous forward pass) |
| `quit` | Exit the loop |

An important, easy-to-miss design detail: **all diagnostic logging goes to
stderr** (the `log()` helper), never to stdout. stdout is reserved
*exclusively* for UCI protocol replies — a GUI that received unexpected
text on stdout would fail to parse the engine's responses. stdout is also
explicitly flushed after every command, ensuring the GUI sees responses
immediately rather than waiting on OS-level output buffering.

---

## 11. Files produced by the pipeline

| Path (constant) | Contents |
|---|---|
| `chesslm_vocab.json` (`VOCAB_PATH`) | Token → ID vocabulary, shared by both stages |
| `chesslm_data_s1_train.pt` / `_s1_val.pt` | Packed token streams, Stage 1 train/val |
| `chesslm_data_s2_train.pt` / `_s2_val.pt` | Packed token streams, Stage 2 train/val |
| `chesslm_stage1_best.pt` (`S1_MODEL_PATH`) | Best Stage-1 (pretraining) weights |
| `chesslm_stage2_best.pt` (`S2_MODEL_PATH`) | Best Stage-2 (finetuning) weights |
| `chesslm_model.pt` (`MODEL_PATH`) | Final deployable weights (copy of the Stage-2 result) |

---

## 12. Hyperparameter reference

| Category | Name | Value |
|---|---|---|
| Architecture | Context length | 350 |
| | Model width | 1024 |
| | Attention heads | 16 |
| | Layers | 12 |
| | Feed-forward width | 4096 |
| | Dropout | 0.15 |
| Training (shared) | Logical batch size | 1024 |
| | Physical batch size | 256 |
| | Gradient accumulation steps | 4 |
| | Gradient clip norm | 1.0 |
| | Weight decay | 0.15 |
| | Min. learning rate | 1e-5 |
| | Validation fraction | 0.03 |
| | Eval interval (steps) | 150 |
| | Early-stop patience | 5 evaluations |
| Stage 1 | ELO range | `[1800, 2400)` |
| | Epochs | 8 |
| | Peak LR | 4e-4 |
| | Warmup steps | 200 |
| Stage 2 | ELO range | `≥ 2400` |
| | Epochs | 10 |
| | Peak LR | 8e-5 |
| | Warmup steps | 200 |
| Inference | Temperature | 0.0 (greedy) |

---

## 13. How to run it

### Requirements

```
torch
numpy
python-chess
datasets   # Hugging Face datasets, for streaming the training data
```

A CUDA GPU is strongly recommended for training (a ~150M-parameter
Transformer trained on millions of move tokens is impractical on CPU); the
script auto-selects the GPU with the most free memory if multiple are
visible, and gracefully falls back to CPU otherwise.

### First run (training from scratch)

#### Run with temperature 0.0 (deterministic)
```sh
./chesSLM
```

#### Run with temperature 0.3
```sh
./chesSLM --temperature 0.3
```

On a machine with no cached artifacts, this will: stream and filter the
dataset, build the vocabulary, tokenize and cache everything, train Stage
1, train Stage 2, save the final model, warm up the model, and then start
listening on stdin for UCI commands.

### Subsequent runs (serving only)

Once `chesslm_model.pt` and `chesslm_vocab.json` exist, simply running
`python chesslm.py` again skips all of the above and immediately loads the
trained model, runs the brief warmup pass, and starts a UCI engine, ready
to be pointed at by any UCI-capable GUI (configure the GUI to use
`python chesslm.py` as the engine's executable).
