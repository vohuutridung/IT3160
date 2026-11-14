"""Turn a policy network into a move-picking agent (single forward pass, no search)."""

from __future__ import annotations

import chess
import numpy as np
import torch
import torch.nn.functional as F

from IT3160.data.action_encoding import decode_move, legal_action_ids
from IT3160.data.canonical import encode_board


class PolicyAgent:
    def __init__(self, model: torch.nn.Module, device: torch.device | str = "cpu"):
        self.model = model.eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def policy(self, board: chess.Board) -> list[tuple[chess.Move, float]]:
        """Legal moves with their probabilities, most likely first."""
        actions = legal_action_ids(board)
        if not actions:
            return []
        x = torch.from_numpy(encode_board(board)).float().unsqueeze(0).to(self.device)
        logits = self.model(x).float().squeeze(0)
        index = torch.tensor(actions, device=logits.device)
        probs = F.softmax(logits[index], dim=0).cpu().numpy()
        order = np.argsort(-probs, kind="stable")
        return [(decode_move(board, actions[i]), float(probs[i])) for i in order]

    def choose(
        self,
        board: chess.Board,
        temperature: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> chess.Move:
        """Argmax when ``temperature == 0``; otherwise sample from p^(1/T)."""
        policy = self.policy(board)
        if not policy:
            raise ValueError("No legal moves in this position")
        if temperature <= 0:
            return policy[0][0]
        weights = np.array([p for _, p in policy], np.float64) ** (1.0 / temperature)
        rng = rng or np.random.default_rng()
        return policy[int(rng.choice(len(policy), p=weights / weights.sum()))][0]
