import torch
from typing import Any

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: Any
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)

def load_checkpoint(
        src: Any,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src, map_location="cpu")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint[["iteration"]]