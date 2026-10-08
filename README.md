# transformer-lm-from-scratch

**A decoder-only Transformer language model with its own optimizer, BPE tokenizer, training loop, and benchmarking script, written in PyTorch without high-level library components.** The code comes from the assignments of Stanford CS336, *Language Modeling From Scratch*, and has been used with TinyStories and OpenWebText.

Licensed under MIT.

### Repository Layout

| Path | Contents |
|---|---|
| `train.py` | Training loop |
| `generate.py` | Text generation from a checkpoint |
| `benchmark.py` | Latency and throughput measurement |
| `src/model.py` | Model layers, loss, AdamW, gradient clipping, LR schedule |
| `src/tokenizer.py` | BPE tokenizer |
| `src/data.py` | Data loading |
| `src/decoding.py` | Decoding routines for generation |
| `src/checkpoint.py` | Checkpoint save and load |
| `utils/train_bpe_tokenizer.py` | Trains the BPE vocabulary |
| `utils/dataset_tokenization.py` | Tokenizes a corpus |
| `data/` | Datasets and tokenizer files |

## Key Features

**Model**
- Pre-norm decoder-only Transformer.
- SwiGLU feed-forward network.
- Rotary positional embeddings.
- Causal multi-head self-attention.
- Hand-written `Linear`, `Embedding`, `RMSNorm`, `softmax`, and `cross_entropy_loss`.

**Optimization**
- Custom AdamW.
- Learning-rate warmup followed by decay.
- Gradient clipping.

**Tokenization**
- Custom byte-pair encoding tokenizer.
- Separate scripts for vocabulary training and corpus tokenization.

**Training and Inference**
- Checkpoint saving and resumption.
- Optional validation set.
- Temperature and top-p sampling.

**Benchmarking**
- Three modes: `forward`, `forward_backward`, `full`.
- Timings bracketed by `torch.cuda.synchronize()`.
- Mean and standard deviation across timed steps.

## Technical Workflow

### Data Preparation

1. `utils/train_bpe_tokenizer.py` trains the BPE vocabulary and merges.
2. `utils/dataset_tokenization.py` encodes the corpus into `.npy` files.

### Training

1. Parse arguments and build `TransformerLM` and AdamW.
2. Restore state from `--resume_from`, if given.
3. For each iteration up to `--max_iters`:
   1. Set the learning rate with `learning_rate_schedule`.
   2. Load a batch of `--context_length` tokens.
   3. Run the forward pass in bfloat16 mixed precision and compute `cross_entropy_loss`.
   4. Backpropagate, clip gradients to `--grad_clip`, and step the optimizer.
4. Log every `--log_interval` iterations; evaluate on the validation set every `--eval_interval` iterations.
5. Save checkpoints to `--checkpoint_path`.

### Generation

1. Load the tokenizer from `data/tokenizer_vocab.pkl` and `data/tokenizer_merges.pkl`.
2. Load the model from `--checkpoint_path`.
3. Encode the prompt and sample up to `--max_new_tokens` tokens with temperature scaling and top-p filtering.
4. Decode the tokens to text.

### Benchmarking

1. Build a model from the given configuration.
2. Run `--warmup_steps` untimed steps.
3. Run `--num_steps` timed steps, synchronizing CUDA around each one.
4. Report latency and throughput as mean and standard deviation.

| `--mode` | Work per step |
|---|---|
| `forward` | Forward pass |
| `forward_backward` | Forward and backward pass |
| `full` | Forward, backward, and optimizer step |

## Installation & Quick Start

**Environment used:** Python 3.11.16, PyTorch 2.14.0+cu130, NumPy 2.4.6, einops 0.8.2. `argparse` ships with the standard library.

```bash
git clone https://github.com/theaidenmax/transformer-lm-from-scratch.git
cd transformer-lm-from-scratch

python -m venv .venv
source .venv/bin/activate

pip install torch numpy==2.4.6 einops==0.8.2
```

An NVIDIA GPU is required for training and benchmarking (`--device` defaults to `cuda`). `generate.py` falls back to CPU when CUDA is unavailable.

Quick start:

```bash
python utils/train_bpe_tokenizer.py
python utils/dataset_tokenization.py
python train.py --train_data_path data/train.npy
```

## Usage Examples

### Tokenizer and Dataset

Both scripts take no command-line arguments.

```bash
python utils/train_bpe_tokenizer.py
python utils/dataset_tokenization.py
```

### Training

```bash
python train.py \
  --train_data_path data/train.npy \
  --val_data_path data/val.npy \
  --checkpoint_path checkpoints/model.pt
```

Resume from a checkpoint:

```bash
python train.py \
  --train_data_path data/train.npy \
  --resume_from checkpoints/model.pt
```

| Argument | Default | Description |
|---|---|---|
| `--train_data_path` | required | Training data (`.npy`) |
| `--val_data_path` | `None` | Validation data; skipped when unset |
| `--checkpoint_path` | `checkpoints/model.pt` | Checkpoint output |
| `--resume_from` | `None` | Checkpoint to resume from |
| `--vocab_size` | `32000` | Vocabulary size |
| `--context_length` | `256` | Sequence length |
| `--d_model` | `512` | Model width |
| `--num_layers` | `6` | Transformer blocks |
| `--num_heads` | `8` | Attention heads |
| `--d_ff` | `2048` | Feed-forward dimension |
| `--batch_size` | `64` | Sequences per batch |
| `--max_iters` | `5000` | Training iterations |
| `--max_lr` | `6e-4` | Peak learning rate |
| `--min_lr` | `6e-5` | Final learning rate |
| `--warmup_iters` | `500` | Warmup iterations |
| `--weight_decay` | `0.1` | AdamW weight decay |
| `--grad_clip` | `1.0` | Gradient clipping threshold |
| `--log_interval` | `10` | Iterations between log lines |
| `--eval_interval` | `500` | Iterations between evaluations |
| `--device` | `cuda` | Compute device |

### Generation

The architecture arguments must match the checkpoint being loaded. Their defaults describe a smaller model than the `train.py` defaults.

```bash
# Default small checkpoint
python generate.py --prompt "Once upon a time"

# Checkpoint trained with train.py defaults
python generate.py \
  --checkpoint_path checkpoints/model.pt \
  --context_length 256 --d_model 512 --num_layers 6 --num_heads 8 --d_ff 2048 \
  --temperature 0.7 --top_p 0.95 --max_new_tokens 200
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint_path` | `checkpoints/tiny_model.pt` | Checkpoint to load |
| `--prompt` | `Once upon a time` | Input text |
| `--max_new_tokens` | `100` | Tokens to generate |
| `--temperature` | `0.8` | Sampling temperature |
| `--top_p` | `0.9` | Nucleus sampling threshold |
| `--device` | `cuda` | Compute device; CPU if CUDA is unavailable |
| `--vocab_size` | `32000` | Vocabulary size |
| `--context_length` | `128` | Sequence length |
| `--d_model` | `256` | Model width |
| `--num_layers` | `4` | Transformer blocks |
| `--num_heads` | `4` | Attention heads |
| `--d_ff` | `1024` | Feed-forward dimension |

### Benchmarking

```bash
python benchmark.py                      # forward, backward, optimizer step
python benchmark.py --mode forward       # forward pass only
python benchmark.py --mode forward_backward --num_steps 20
```

| Argument | Default | Description |
|---|---|---|
| `--d_model` | `512` | Model width |
| `--num_layers` | `4` | Transformer blocks |
| `--num_heads` | `16` | Attention heads |
| `--d_ff` | `1344` | Feed-forward dimension |
| `--vocab_size` | `10000` | Vocabulary size |
| `--context_length` | `256` | Sequence length |
| `--batch_size` | `32` | Sequences per batch |
| `--warmup_steps` | `5` | Untimed steps |
| `--num_steps` | `10` | Timed steps |
| `--mode` | `full` | `forward`, `forward_backward`, or `full` |
