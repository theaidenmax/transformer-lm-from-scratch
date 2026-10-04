import torch
from typing import Any, Optional
from src.model import Optimizer

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
        optimizer: Optional[Optimizer] = None,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint["iteration"]