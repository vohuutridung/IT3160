"""Load a trained policy network from a checkpoint without hard-coding its size."""

from __future__ import annotations

from pathlib import Path
import re

import torch

from IT3160.models.resnet import ChessPolicyResNet


def infer_model_config(state_dict: dict) -> dict:
    """Recover constructor arguments of ChessPolicyResNet from its weights."""
    try:
        conv_in = state_dict["conv_in.weight"]
        policy = state_dict["policy_conv.weight"]
    except KeyError as exc:
        raise ValueError(f"Not a ChessPolicyResNet state dict (missing {exc})") from None
    block_ids = {
        int(match[1]) for key in state_dict
        if (match := re.fullmatch(r"blocks\.(\d+)\.conv1\.weight", key))
    }
    if block_ids != set(range(len(block_ids))):
        raise ValueError("Residual block indices are not contiguous")
    return {
        "in_channels": int(conv_in.shape[1]),
        "channels": int(conv_in.shape[0]),
        "num_blocks": len(block_ids),
        "policy_planes": int(policy.shape[0]),
    }


def load_policy_model(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[ChessPolicyResNet, dict]:
    """Return ``(model in eval mode, info)``.

    Accepts the checkpoints written by ``scripts/train.py`` or a bare state dict.
    ``info`` holds the model config, the epoch and the training config when present.
    """
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict, checkpoint = checkpoint, {}
    config = checkpoint.get("model_config") or infer_model_config(state_dict)
    model = ChessPolicyResNet(**config)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    info = {
        "checkpoint": str(path),
        "model_config": config,
        "parameters": model.count_parameters(),
        "epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "train_config": checkpoint.get("config"),
    }
    return model, info
