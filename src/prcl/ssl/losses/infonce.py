"""InfoNCE / NT-Xent loss for contrastive SSL."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.

    Standard SimCLR contrastive loss. For a batch of N samples producing
    2N augmented views, each positive pair is (2i, 2i+1) and all other
    2(N-1) samples in the batch serve as negatives.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss.

        Args:
            z: Projected features of shape (2N, proj_dim) where z[2i] and z[2i+1]
               are the two views of sample i.

        Returns:
            Scalar loss.
        """
        batch_size = z.shape[0] // 2  # N
        z = F.normalize(z, dim=1)

        # Full 2N x 2N similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity on the diagonal
        mask_self = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask_self, -9e15)

        # Positive pair indices: (2i, 2i+1) and (2i+1, 2i)
        pos_indices = torch.arange(2 * batch_size, device=z.device)
        pos_indices[0::2] += 1  # 2i -> 2i+1
        pos_indices[1::2] -= 1  # 2i+1 -> 2i

        # Numerator: similarity of positive pairs
        pos_sim = sim[torch.arange(2 * batch_size, device=z.device), pos_indices]

        # Denominator: sum over all non-self similarities (log-sum-exp)
        loss = -pos_sim + torch.logsumexp(sim, dim=1)

        return loss.mean()
