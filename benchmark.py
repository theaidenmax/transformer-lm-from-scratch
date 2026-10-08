import argparse
import timeit
import torch
import numpy as np
from src.model import TransformerLM, AdamW, cross_entropy

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Benchmarking")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["forward", "forward_backward", "full"],
        default="full",
    )
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()

def run_benchmark():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    def run_step():
        if args.mode == "forward":
            with torch.no_grad():
                logits = model(x)
        elif args.mode == "forward_backward":
            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
        elif args.mode == "full":
            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

    for _ in range(args.warmup_steps):
        run_step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    timings = []

    torch.cuda.memory._record_memory_history(max_entries=1000000)

    for _ in range(args.num_steps):
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = timeit.default_timer()

        run_step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        t1 = timeit.default_timer()
        timings.append(t1 - t0)

    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    mean_time = np.mean(timings)
    std_time = np.std(timings)

    print(f"Mode: {args.mode}")
    print(f"Warmup steps: {args.warmup_steps}, Measured steps: {args.num_steps}")
    print(f"Average time per step: {mean_time * 1000:.3f} ms")
    print(f"Standard deviation: {std_time * 1000:.3f} ms")

if __name__ == "__main__":
    run_benchmark()