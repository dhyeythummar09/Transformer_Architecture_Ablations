# Transformer Architecture Ablation Lab

Implemented and benchmarked five encoder-decoder Transformer variants from scratch in PyTorch, comparing positional encoding, attention, normalization, and byte-level modeling.

## Highlights

- Built the core encoder-decoder Transformer stack manually without `nn.Transformer` or `nn.MultiheadAttention`
- Implemented custom scaled dot-product attention, Multi-Head Attention (MHA), and Grouped-Query Attention (GQA)
- Compared sinusoidal positional encoding with RoPE
- Compared LayerNorm with RMSNorm
- Implemented custom BPE tokenizers from scratch for binary ciphertext and plaintext
- Implemented a byte-level latent Transformer pipeline with local 4-byte patches
- Trained all variants under a controlled experimental setup
- Logged training and validation metrics with Weights & Biases
- Evaluated exact sequence accuracy, bit accuracy, Levenshtein distance, BLEU, and ROUGE-L
- Added a source-shuffling diagnostic to verify that the decoder actually depends on the ciphertext input

## Architecture Variants

Each configuration changes one main component relative to the baseline.

| Config | Positional Encoding | Attention | Normalization | Input Representation |
|---|---|---|---|---|
| **C1** | Sinusoidal | MHA | LayerNorm | Learned BPE |
| **C2** | RoPE | MHA | LayerNorm | Learned BPE |
| **C3** | Sinusoidal | GQA | LayerNorm | Learned BPE |
| **C4** | Sinusoidal | MHA | RMSNorm | Learned BPE |
| **C5** | Sinusoidal | MHA | LayerNorm | Raw bytes + 4-byte local patches |

## Results

Primary evaluation uses greedy decoding on 5,487 independent test chunks.

| Config | Sequence Accuracy | Bit Accuracy | Mean Levenshtein | BLEU | ROUGE-L |
|---|---:|---:|---:|---:|---:|
| C1 | 46.75% | 92.51% | 1.78 | 81.93 | 92.03% |
| C2 | 45.47% | 92.65% | 1.83 | 81.20 | 91.75% |
| C3 | 43.12% | 92.35% | 2.03 | 80.13 | 91.13% |
| C4 | 46.07% | 92.90% | 1.76 | 81.70 | 91.89% |
| **C5** | **99.91%** | **100.00%** | **0.00** | N/A | N/A |

Training and validation efficiency:

| Config | Training Time | Peak GPU Memory | Best Val Loss | Val Token Accuracy |
|---|---:|---:|---:|---:|
| C1 | 31.1 min | 1166 MB | 1.0479 | 91.33% |
| C2 | 32.9 min | 1171 MB | 1.0527 | 91.31% |
| C3 | 25.4 min | 1153 MB | 1.0921 | 90.41% |
| **C4** | **23.2 min** | 1084 MB | 1.0555 | 91.08% |
| **C5** | 27.9 min | **420 MB** | **0.4865** | **100.00%** |

### Main Takeaways

- **RoPE (C2)** behaved similarly to sinusoidal positional encoding on these relatively short chunked sequences.
- **GQA (C3)** reduced training time but slightly reduced reconstruction quality, showing an efficiency-quality tradeoff.
- **RMSNorm (C4)** gave the strongest BPE-based efficiency result and the highest BPE bit accuracy.
- **Byte-level latent modeling (C5)** produced the largest improvement: nearly perfect reconstruction with substantially lower GPU memory usage.
- In this experiment, **input representation had a much larger effect than the other single architectural changes**.

## Data Pipeline

The task maps encrypted binary sequences back to plaintext.

The original aligned plaintext-ciphertext pairs are split into train/validation/test sets **before chunking** to avoid leakage.

Each plaintext line is divided into non-overlapping 64-character ASCII chunks:

```text
64 plaintext characters
        ↕
512 ciphertext bits
```

The final shorter chunk of each line is retained.

Resulting chunk counts:

```text
Train:      38,257
Validation:  5,406
Test:        5,487
```

### C1-C4: Custom BPE

Two custom BPE tokenizers are trained from scratch:

```text
Ciphertext vocabulary: 2048
Plaintext vocabulary:  4096
```

The ciphertext tokenizer learns directly over binary symbols `0` and `1`.

### C5: Byte-Level Pipeline

C5 does not use learned subword segmentation.

```text
8 ciphertext bits → 1 byte
4 bytes → 1 local patch
64 bytes → 16 global patch representations
```

A lightweight local Transformer encodes each 4-byte patch before the global Transformer. A corresponding local decoder reconstructs output bytes.

## Model Configuration

Common settings:

```text
d_model:             256
attention heads:     8
encoder layers:      4
decoder layers:      4
feed-forward size:   1024
dropout:             0.1
max sequence length: 1024
```

C3 uses:

```text
Query heads:      8
Key/value heads:  2
```

C5 uses:

```text
Patch size:       4 bytes
Local dimension:  128
Local heads:      4
Local layers:     1
```

## Training Setup

```text
Optimizer:          AdamW
Learning rate:      3e-4
Betas:              (0.9, 0.98)
Weight decay:       0.01
Batch size:         64
Label smoothing:    0.05
Gradient clipping:  1.0
Training steps:     25,000
Warmup steps:       2,000
Minimum LR:         3e-5
Precision:          FP16
Seed:               42
```

The learning rate uses linear warmup followed by cosine decay.

Experiments were run on an NVIDIA GeForce RTX 3050 Laptop GPU.

## Source-Shuffling Diagnostic

To verify that the decoder was not simply behaving like a plaintext language model, the validation source examples were shuffled while targets were kept fixed.

For the C1 baseline:

```text
Normal validation loss:          1.0288
Shuffled-source validation loss: 11.2233
Shuffle penalty:                10.1945
```

The large increase in loss indicates that the model strongly depends on the ciphertext input.

## Repository Structure

```text
.
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── utils.py
│   └── models/
│       ├── attention.py
│       ├── positional.py
│       ├── norm.py
│       └── blt.py
│
├── outputs/
│   ├── cipher_tokenizer.json
│   ├── tokenizer.json
│   ├── metrics_c1.json
│   ├── metrics_c2.json
│   ├── metrics_c3.json
│   ├── metrics_c4.json
│   └── metrics_c5.json
│
├── quick_check.py
├── run_experiments.py
├── requirements.txt
├── Writeup_A1_ANLP.pdf
└── README.md
```

### File Map

| File | Purpose |
|---|---|
| `src/dataset.py` | Data loading, splitting, aligned chunking, custom BPE, byte tokenizer, datasets and collators |
| `src/models/attention.py` | Scaled dot-product attention, MHA and GQA |
| `src/models/positional.py` | Sinusoidal positional encoding and RoPE |
| `src/models/norm.py` | LayerNorm and RMSNorm |
| `src/models/blt.py` | Local byte encoder/decoder and patch-level Transformer components |
| `src/train.py` | Configuration profiles, Transformer encoder/decoder, training loop, validation, scheduler, checkpointing and W&B logging |
| `src/utils.py` | Greedy decoding evaluation, metrics, full-line reconstruction and result export |
| `quick_check.py` | Lightweight sanity checks |
| `run_all_rtx.py` | Convenience script for running the full experiment set |

## Installation

Python 3.10+ is recommended.

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd transformer-architecture-ablation-lab

pip install -r requirements.txt
```

For GPU training, install a CUDA-compatible PyTorch build for your system.

## Dataset

The original dataset is not included in the public repository.

Place the aligned files at:

```text
data/
├── brown_cipher.txt
└── brown_plain.txt
```

Each line in `brown_cipher.txt` must correspond to the same line in `brown_plain.txt`.

## Training

Train a single configuration:

```bash
python -m src.train --config_name C1 --source_shuffle_diagnostic --use_wandb
```

Replace `C1` with `C2`, `C3`, `C4`, or `C5`.

Example:

```bash
python -m src.train --config_name C5 --source_shuffle_diagnostic --use_wandb
```

## Evaluation

Evaluate all five trained configurations:

```bash
python -m src.utils \
  --configs C1 C2 C3 C4 C5 \
  --eval_batch_size 64 \
  --amp_dtype fp16
```

Evaluation outputs include:

```text
outputs/metrics_c1.json
outputs/metrics_c2.json
outputs/metrics_c3.json
outputs/metrics_c4.json
outputs/metrics_c5.json
```

The implementation also stores per-example predictions and reconstructs original full lines as a secondary diagnostic.

## Pretrained Checkpoints

Pretrained C1-C5 checkpoints and tokenizers are hosted on Hugging Face:

[**Hugging Face — Transformer Ablation Checkpoints**](https://huggingface.co/dhyeythummar9/ANLP-A1-Transformer-Ablations)

Large checkpoint files are intentionally not stored directly in this GitHub repository.

## Experiment Tracking

Training and validation metrics were logged with Weights & Biases, including:

- training loss
- validation loss
- validation token accuracy
- learning rate
- gradient norm
- GPU memory
- training throughput
- source-shuffling diagnostics

> W&B run visibility depends on the workspace permissions associated with the original experiments.

## Implementation Notes

The project intentionally avoids PyTorch's high-level Transformer modules for the core architecture.

Implemented manually:

- scaled dot-product attention
- multi-head attention
- grouped-query attention
- sinusoidal positional encoding
- rotary positional encoding
- LayerNorm
- RMSNorm
- position-wise feed-forward networks
- encoder and decoder blocks
- masking and causal decoding
- custom BPE tokenization
- byte-level patch encoder/decoder
- warmup + cosine learning-rate scheduling
- greedy autoregressive decoding

## Experiments Report 
[`Report.pdf`](./Report.pdf)

The full experimental write-up is available in:

[`Writeup_A1_ANLP.pdf`](./Writeup_A1_ANLP.pdf)

## Reproducibility

All five variants were trained with the same core hyperparameters wherever applicable, with a fixed seed of `42`.

C2-C5 each modify one primary component relative to C1, enabling controlled architectural comparison.

---

If you use this repository as a reference, feel free to cite or link back to it.
