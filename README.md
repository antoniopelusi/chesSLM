# ChesSLM

**ChesSLM** is a ~150M parameter, decoder-only causal language model built entirely from scratch in PyTorch to play chess. By treating chess matches as a textual language modeling problem, the network learns to process tokenized sequences of Standard Algebraic Notation (SAN) moves and predict the most stylistically and tactically accurate next move.

The architecture contains no high-level transformer abstractions (e.g., Hugging Face Transformers, PyTorch Lightning); every element—from custom vocabulary tokenization and multi-head causal attention blocks to the two-stage curriculum training loop—is designed and executed natively from the ground up.

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
Pull the production weights and vocabulary configuration directly from the remote Hugging Face repository:
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

## 🛠️ Core Technologies & "From-Scratch" Stack

* **Deep Learning Framework:** `PyTorch (v2.x+)` utilized natively. The architecture scales out using PyTorch's specialized `torch.compile()` Inductor backend to fuse kernels and speed up inference/training loops.
* **Core Transformer Mechanics:** Engineered entirely within `model.py`. Features a custom `CausalSelfAttention` module using scaled dot-product attention (`torch.nn.functional.scaled_dot_product_attention`), which natively leverages hardware-accelerated **FlashAttention** kernels when backed by compatible CUDA compute capabilities.
* **Weight Tying:** Embedded an optimization strategy where the input token embedding matrix weights are tied directly to the output linear projection layer (`self.head.weight = self.tok_emb.weight`), significantly compressing the model's footprint while stabilizing the linguistic representation space.
* **State Verification & Move Masking:** Coupled with the `python-chess` library to govern rules validation during inference. This forms a hybrid neuro-symbolic bridge: the neural network handles probabilistic move generation, while a programmatic legal-move mask filters out invalid token predictions, guaranteeing 100% rule-compliant engine behavior.

---

## 🧠 Deep Dive: End-to-End Training Flow

The complete training architecture, implemented in `train.py`, operates as an automated multi-stage pipeline designed to scale modern text generation concepts into structural board game logic.

```
[Hugging Face Stream] ──> [ELO Filter & SAN Normalization] ──> [Vocabulary Synthesis]
                                                                        │
┌───────────────────────────────────────────────────────────────────────┘
│
├──> [STAGE 1: Pre-training] ──> ELO [1800, 2400)  ──> Learns Rules, Openings & Tactics
│                                                                 │
└──> [STAGE 2: Fine-tuning]  ──> ELO >= 2400       ──> Learns Positional Grandmaster Strategy
```

### Step 1: Ingestion, Filtration & Vocabulary Synthesis

1. **Streaming Data Ingestion:** The dataset (`angeluriot/chess_games`) is streamed iteratively to eliminate memory-bound bottlenecks.
2. **Filtration Pass:** Games are discarded if they contain incomplete metadata or span fewer than 10 total moves. The white and black player ratings are cross-examined to derive a baseline minimum ELO rating for curriculum classification.
3. **Move Sanitization:** Move strings are parsed, stripping check (`+`) and checkmate (`#`) modifiers to reduce token vocabulary inflation while preserving structural move definitions.
4. **Tokenization Strategy:** A vocabulary mapping dictionary is generated dynamically from the training pool. Special tokens are introduced to manage causal boundaries:
* `[PAD]`: Sequences shorter than 350 moves are padded to fit a fixed contextual width.
* `[BOS]`: Prepended to mark the initiation of a fresh game context.
* `[EOS]`: Appended to signal game termination.



### Step 2: Two-Stage Curriculum Strategy

Instead of optimizing indiscriminately over the entire dataset, the network undergoes a strict chronological ELO-based curriculum:

* **Stage 1 (Pre-training — 8 Epochs):** Constrained to games where the baseline ELO resides within the `[1800, 2400)` band. This phase optimizes high-entropy distributions, instructing the network on foundational game structure, common opening books, and basic tactical calculations.
* **Stage 3 (Fine-tuning — 10 Epochs):** Shunted exclusively to elite Grandmaster/Master games where ratings exceed `2400`. The learning rate is dramatically throttled down to subtly shift the attention weights toward high-level positional configurations, end-game execution, and nuanced strategic play without suffering catastrophic forgetting.

### Step 3: The Forward-Backward Optimization Mechanics

During active training execution, each step undergoes the following performance and mathematical routines:

* **Batching & Gradient Accumulation:** To simulate a highly stable macro batch size of `1024` sequences on restricted hardware configurations, gradients are mathematically accumulated across consecutive physical sub-batches of `256` items over `4` accumulation steps before triggering an optimizer step.
* **Automatic Mixed Precision (AMP):** Execution frames are dynamically cast using PyTorch's native `autocast` context manager. The system checks hardware compatibility at boot, prioritizing `bfloat16` for numerical stability, or falling back onto `float16` scaled via an automated `GradScaler` to prevent underflow in backward weight updates.
* **Target Shifting & Loss Computation:** Input token vectors are offset by one timestep relative to target sequences. The cross-entropy loss function is evaluated across the sequence length, completely ignoring loss computations on padding index blocks via `ignore_index=pad_id`.
* **Split Weight Decay Optimization:** The network parameters are split into two decoupled optimizer groups:
1. All 2D weight matrices are penalized with a structural weight decay coefficient of `0.15` to strictly counter overfitting.
2. All 1D tensors (biases, embedding parameters, and embedding scaling parameters in LayerNorm blocks) are fully exempt from weight decay.


* **Learning Rate Scheduling:** The effective learning rate follows a strict Cosine Decay schedule. It starts with a standard linear warmup over `200` initialization steps up to the peak stage-specific learning rate, subsequently executing a cosine curve degradation down to a hard floor value of `1e-5`.
* **Gradient Clipping:** Prior to executing optimizer updates, the L2 norm of the accumulated gradients is checked and capped at a hard threshold of `1.0` to eliminate exploding gradient steps through dense attention blocks.

---

## 📁 Repository Blueprint

* `chesSLM.py`: System operational gateway. Validates local environment parameters, triggers GPU load-balancing heuristics, interprets CLI directives, and launches runtime instances.
* `config.py`: Hardcoded model dimension parameters, sequence length constraints (`350`), training hyper-parameters, and mutable inference options.
* `model.py`: Natively engineered Transformer architecture blocks, LayerNorm pipelines, and causal mask tensors.
* `data.py` / `vocab.py`: Data pipeline logic, HF dataset download routines, array caching, split generation, and vocabulary lookup tables.
* `train.py`: Rigid optimization loops, validation checkpoint management, AMP context setups, and curriculum execution logic.
* `inference.py`: Real-time text generation routing, temperature sampling calculations, and full UCI protocol command parsing loops.
* `util.py`: Subprocess interfaces mapping system hardware capabilities (`nvidia-smi` parser).
* `paths.py`: Isolated global file system paths and remote repository registry IDs.
