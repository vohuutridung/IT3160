"""ResNet-16x256 Policy Network for chess move prediction with legal move masking."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual Block with two 3x3 convolutions, BatchNorm and ReLU."""

    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual, inplace=True)
        return out


class ChessPolicyResNet(nn.Module):
    """Supervised Chess Policy Network based on AlphaZero-style ResNet backbone.

    Attributes:
        in_channels: 17 planes (12 piece planes, 4 castling planes, 1 en-passant plane).
        num_blocks: 16 residual blocks.
        channels: 256 filters per residual block.
        action_space: 4672 (8x8x73 action planes).
    """

    def __init__(
        self,
        in_channels: int = 17,
        num_blocks: int = 16,
        channels: int = 256,
        policy_planes: int = 73,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.channels = channels
        self.policy_planes = policy_planes
        self.action_space = 8 * 8 * policy_planes  # 4672

        # Initial representation layer
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)

        # 16 Residual Blocks
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # Policy Head: 1x1 conv mapping 256 -> 73 planes
        self.policy_conv = nn.Conv2d(channels, policy_planes, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input board tensor [B, 17, 8, 8], float32.
            legal_mask: Optional boolean tensor [B, 4672] with True for legal moves.

        Returns:
            logits: [B, 4672] move logits (masked if legal_mask is provided).
        """
        out = F.relu(self.bn_in(self.conv_in(x)), inplace=True)
        for block in self.blocks:
            out = block(out)

        # Policy convolution: [B, 73, 8, 8]
        policy_out = self.policy_conv(out)
        logits = policy_out.view(x.size(0), -1)  # [B, 4672]

        if legal_mask is not None:
            # Mask illegal moves with a large negative value
            # Safe value for both float16 (-1e4) and float32 (-1e9)
            neg_inf = -1e4 if logits.dtype == torch.float16 else -1e9
            logits = logits.masked_fill(~legal_mask, neg_inf)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
