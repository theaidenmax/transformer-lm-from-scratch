import argparse
import math
import os
import time
import torch
import torch.nn as nn
import numpy as np

from src.model import TransformerLM, Optimizer, AdamW, gradient_clipping, cross_entropy, learning_rate_schedule
from src.data import data_loading
from src.checkpoint import save_checkpoint, load_checkpoint

def estimate_loss(model, data, batch_size, context_length, device, eval_iters=50):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = data_loading(data, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
        model.train()
        return float(np.mean(losses))

def train(args):
    device = args.device if torch.cuda.is_available() or args.device == "mps" else "cpu"
    print(f"Running on: {device}")

    train_data = np.load(args.train_data_path, mmap_mode="r")
    val_data = np.load(args.val_data_path, mmap_mode="r") if args.val_data_path else None

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

    start_iter = 0
    if args.resume_from:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed training procces from iteration: {start_iter}")

    model.train()

    start_time = None
    WARMUP_STEPS = 10

    for iter_num in range(start_iter, args.max_iters):
        if iter_num == start_iter + WARMUP_STEPS:
            torch.cuda.synchronize()
            start_time = time.time()

        lr = learning_rate_schedule(iter_num, args.max_lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x_batch, y_batch = data_loading(train_data, args.batch_size, args.context_length, device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x_batch)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        # logits = model(x_batch)
        # loss = cross_entropy(logits.view(-1, logits.size(-1)), y_batch.view(-1))

        optimizer.zero_grad()
        loss.backward()

        gradient_clipping(model.parameters(), max_norm=args.grad_clip)

        optimizer.step()

        if iter_num % args.log_interval == 0:
            sec_per_step_str = "N/A"
        
        if iter_num > start_iter + WARMUP_STEPS and start_time is not None:
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            steps_done = iter_num - (start_iter + WARMUP_STEPS)
            sec_per_step = elapsed / steps_done
            sec_per_step_str = f"{sec_per_step:.4f}s"

        print(f"Iter {iter_num}/{args.max_iters} Loss: {loss.item():.4f} LR: {lr:.6f} Sec/Step: {sec_per_step_str}")

    if iter_num > 0 and iter_num % args.eval_interval == 0:
        if val_data is not None:
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device)
            print(f"Iter {iter_num} Val Loss: {val_loss:.4f}")

        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        save_checkpoint(model, optimizer, iter_num, args.checkpoint_path)
        print(f"Checkpoint saved in {args.checkpoint_path}")

    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    save_checkpoint(model, optimizer, args.max_iters, args.checkpoint_path)
    print(f"Final checkpoint saved in {args.checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS336 Language Model Training")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/model.pt")
    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--max_lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    train(args)