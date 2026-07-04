# ♟️ chesSLM
Chess **S**mall **L**anguage **M**odel

> Level: Stockfish ~1650 ELO on temperature 0.0

**ChesSLM** is a ~100M parameters, decoder-only causal language model built entirely from scratch in PyTorch to play chess. By treating chess matches as a textual language modeling problem, the network learns to process tokenized sequences of Standard Algebraic Notation (SAN) moves and predict the most stylistically and tactically accurate next move.

The architecture contains no high-level transformer abstractions (e.g., Hugging Face Transformers, PyTorch Lightning); every element—from custom vocabulary tokenization and Rotary Positional Embeddings (RoPE) to the Pre-LN multi-head causal blocks and the two-stage curriculum training loop—is designed and executed natively from the ground up.

---

## 🚀 Setup and Run

### 1. Environment Initialization

Initialize a clean Python virtual environment and upgrade the dependency package manager using the provided automation target:

```bash
make setup
```

### 2. Weight and Artifact Acquisition (One-Time Setup)

To operationalize the engine, you must either download the pre-compiled production weights or execute the training pipeline.

* **Option A: Download Pre-trained Weights (Recommended)**
Pull the production weights and vocabulary configuration directly from the remote Hugging Face repository (`antoniopelusi/chesSLM`):

```bash
./chesSLM.py --getmodel
```

* **Option B: Train From Scratch**
Trigger the local two-stage curriculum training pipeline. This will dynamically stream dataset chunks and build artifacts locally:

```bash
./chesSLM.py --train
```

### 3. Execution & Interface Hooking

ChesSLM implements the Universal Chess Interface (UCI) protocol. At boot, the main entry point runs an automated hardware assessment, dynamically querying `nvidia-smi` to pin execution to the GPU containing the highest available VRAM, and provisions the necessary `LD_LIBRARY_PATH` variables.

* **GUI Integration (Graphical Play):**
You can download CuteChess GUI using the following command:
```bash
make cutechess
```


Then:
1. Navigate to **Tools > Settings > Engines > Add**.
2. Set the executable path pointing to `./chesSLM.py`.
3. Ensure the working directory is explicitly targeted at the repository root.
4. **Engine Options:** Set the `Temperature` parameter via the `Advanced` panel of an engine entry. A value between **0.0 and 0.3** is highly recommended for stable positional play. Setting it to `0.0` automatically routes inference to pure deterministic argmax (greedy decoding).


* **Raw CLI Interactive Session:**
```bash
./chesSLM.py
```


Here you can manually pipe standard UCI instructions (`isready`, `position startpos moves e2e4`, `go`) into stdin.

---

## 🛠️ Core Technologies & Architectural Stack

ChesSLM operates on a hyperparameter configuration optimized for fixed-pool multi-epoch training: a Context Window of **350 tokens**, an Embedding Dimension (`d_model`) of **896**, **14** Attention Heads, **10** Transformer Layers, and a Feed-Forward dimension (`d_ff`) of **3584**, with a baseline network dropout of **0.10**.

The forward pass natively implements the following layer-by-layer architectural flow:

1. **Token Embedding & Regularization:** The input sequence of SAN token IDs is mapped to the 896-dimensional hidden space via a standard `nn.Embedding` matrix. Absolute positional embeddings are entirely omitted. The raw embeddings immediately pass through an `nn.Dropout` layer to prevent early sequence memorization.
2. **Pre-LN Transformer Blocks (x10):** The sequence cascades through a stack of 10 identical decoder-only blocks. Each block strictly adheres to a Pre-Layer Normalization (`nn.LayerNorm`) topology, applying normalization before the attention and feed-forward sub-layers to stabilize gradient flow across the residual stream.
3. **Causal Self-Attention with RoPE:** Inside the attention sub-layer, a single bias-free fused linear projection (`nn.Linear(d_model, 3 * d_model)`) splits the hidden states into Query, Key, and Value tensors. Rather than using absolute positions, **Rotary Positional Embeddings (RoPE)** are applied dynamically: precomputed sine and cosine matrices geometrically rotate the Query and Key vectors in 2-D subspaces. This encodes relative positioning directly into the dot-product, offering superior inductive bias for sequential move logic. The tensors are then passed to `torch.nn.functional.scaled_dot_product_attention` (`is_causal=True`) to leverage hardware-accelerated FlashAttention kernels.
4. **Feed-Forward Network (MLP):** The context-aware vectors exit the attention block, undergo a second Pre-LN, and enter the FFN to process rigid chess logic. The FFN expands the dimension by a factor of 4x (to 3584) via a linear layer, applies a non-linear `nn.GELU` activation, drops out 10% of the weights, and compresses the space back to 896 via a final linear projection.
5. **Residual Scale Initialization:** Mimicking GPT-2 practices, linear layers that write directly onto the residual stream (the attention and FFN output projections) have their standard deviation scaled down by `1 / sqrt(2 * n_layers)` during initialization (`RESIDUAL_SCALE_INIT = True`), compensating for variance accumulation across depth.
6. **Final Normalization & Unembedding (Weight Tying):** The tensor exiting the final transformer block undergoes a concluding `nn.LayerNorm`. It is then projected into the vocabulary distribution via a bias-free linear head. To drastically compress the model footprint and enforce symmetric semantic clustering, this output head utilizes **Weight Tying** (`self.head.weight = self.tok_emb.weight`), sharing its parameter matrix identically with the input token embedding layer.
7. **Neuro-Symbolic Move Masking:** During inference, the neural network generates logits for the entire vocabulary, but a programmatic programmatic legal-move mask (via `python-chess`) filters the distribution prior to temperature scaling and sampling. This hybrid approach guarantees 100% rule-compliant engine behavior.

---

## 🧠 Deep Dive: End-to-End Training Flow

The complete training architecture, implemented in `train.py`, operates as an automated multi-stage pipeline designed to scale modern text generation concepts into structural board game logic.

```text
[Hugging Face Stream] ──> [Validation & SAN Normalization] ──> [ELO Routing] ──> [Vocabulary Synthesis]
                                                                        │
┌───────────────────────────────────────────────────────────────────────┘
│
├──> [STAGE 1: Pre-training] ──> best known ELO < 2400, or unrated ──> Learns Rules, Openings & Tactics
│                                                                 │
└──> [STAGE 2: Fine-tuning]  ──> best known ELO >= 2400          ──> Learns Positional Grandmaster Strategy
```

### Step 1: Ingestion, Filtration & Vocabulary Synthesis

1. **Streaming Data Ingestion:** The dataset (`angeluriot/chess_games`) is streamed iteratively to eliminate memory-bound bottlenecks.
2. **Filtration Pass:** Games shorter than 10 half-moves are discarded. Every remaining move is replayed against a legality checker; any game where a move fails to parse or isn't legal in sequence is dropped as malformed.
3. **ELO Routing:** The higher of the two known player ratings (if any) dictates the routing. Games with a known rating >= 2400 go to the fine-tuning pool; everything else goes to the pre-training pool.
4. **Tokenization Strategy:** A vocabulary mapping dictionary is mined from the SAN move strings (with check/mate modifiers stripped), plus three special tokens:
* `<pad>`: Fills unused positions at inference, and masks target positions in the loss function.
* `<bos>`: Prepended to mark the start of each game.
* `<eos>`: Appended to mark the end of each game.


5. **Orphan Fragment Masking:** Games are concatenated into a flat 1-D stream and sliced into fixed 350-token context windows. If a window begins mid-game, the preceding moves are absent, making the board state un-reconstructable. The `ChessDataset` automatically identifies these "orphan fragments" and sets target tokens prior to the first `<bos>` to the `<pad>` ID, ensuring they are ignored during loss computation.

### Step 2: Two-Stage Curriculum Strategy

Instead of optimizing indiscriminately over the entire dataset, the network undergoes a strict chronological ELO-based curriculum:

* **Stage 1 (Pre-training — up to 5 Epochs):** Trained on every game not routed to Stage 2. This phase optimizes high-entropy distributions, instructing the network on foundational game structure, common opening books, and basic tactical calculations. Peak learning rate: `4e-4`.
* **Stage 2 (Fine-tuning — up to 6 Epochs):** Restricted to games where at least one player has a known rating >= 2400. The learning rate is throttled down (5x lower peak, `8e-5`) to subtly shift the attention weights toward high-level positional configurations, end-game execution, and nuanced strategic play without suffering catastrophic forgetting.

*Note: Epoch counts are ceilings; a validation-based early stopping mechanism typically halts stages before completion.*

### Step 3: The Forward-Backward Optimization Mechanics

During active training execution, each step undergoes the following performance and mathematical routines:

* **Batching & Gradient Accumulation:** To simulate a highly stable macro batch size of `1024` sequences on restricted hardware configurations, gradients are mathematically accumulated across consecutive physical sub-batches of `256` items over `4` accumulation steps before triggering an optimizer step.
* **Automatic Mixed Precision (AMP):** Execution frames are dynamically cast using PyTorch's native `autocast`. The system checks hardware compatibility at boot, prioritizing `bfloat16` (scaler-free) for numerical stability, or falling back onto `float16` scaled via an automated `GradScaler` to prevent gradient underflow.
* **Split Weight Decay Optimization (`AdamW`):** The network parameters are split into two decoupled optimizer groups:
1. All 2D+ weight matrices — including the tied token embedding/output matrix — are penalized with a weight decay coefficient of `0.10`.
2. All 1D tensors (biases and LayerNorm scales) are fully exempt from weight decay.


* **Learning Rate Scheduling:** The effective learning rate follows a strict Cosine Decay schedule. It utilizes a linear warmup over `200` initialization steps up to the peak stage-specific learning rate, subsequently executing a cosine curve degradation down to a hard floor value of `1e-5`.
* **Gradient Clipping:** Prior to executing optimizer updates, the L2 norm of the accumulated gradients is checked and capped at a hard threshold of `1.0` to eliminate exploding gradient steps through dense attention blocks.

---

## 📁 Repository Blueprint

* `chesSLM.py`: System operational gateway. Validates local environment parameters, triggers GPU load-balancing heuristics, interprets CLI directives, and launches runtime instances.
* `config.py`: Hardcoded model dimension parameters, sequence length constraints, training hyper-parameters, and mutable inference options.
* `model.py`: Natively engineered Transformer architecture blocks, RoPE injections, Pre-LN pipelines, multi-head causal attention mechanisms, and custom weight initializers.
* `data.py` / `vocab.py`: Data pipeline logic, HF dataset download routines, context-window slicing (with orphan fragment masking), split generation, and vocabulary lookup tables.
* `train.py`: Rigid optimization loops, validation checkpoint management, AMP context setups, cosine LR scheduling, and curriculum execution logic.
* `inference.py`: Real-time autoregressive text generation, temperature sampling calculations, KV caching simulation, and full UCI protocol command parsing loops.
* `run.py`: Primary execution script. Serves as the main entry point for triggering specific sub-modules, launching the application, or managing the engine's lifecycle.
* `util.py`: Subprocess interfaces mapping system hardware capabilities (`nvidia-smi` parser).
* `paths.py`: Isolated, dependency-free global file system paths and remote HF repository registry IDs utilized by the lightweight downloader.
