import torch


def get_device() -> torch.device:
    """Select the best available compute device.

    Prefers CUDA (NVIDIA), then DirectML (Windows GPUs without CUDA/ROCm
    support, e.g. AMD cards), falling back to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml

        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass

    return torch.device("cpu")
