import numpy as np
import torch

def data_loading(
        x: np.ndarray,
        batch_size: int,
        context_length: int,
        device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_idx = len(x) - context_length
    ix = np.random.randint(0, max_idx, size=batch_size)

    inputs_list = [x[i : i + context_length] for i in ix]
    targets_list = [x[i + 1 : i + context_length + 1] for i in ix]

    inputs = torch.tensor(np.array(inputs_list), dtype=torch.long, device=device)
    targets = torch.tensor(np.array(targets_list), dtype=torch.long, device=device)

    return inputs, targets