"""Probe-Consistency Forensics (PCF) — label-free suspicion scoring for SSL.

PCF computes per-sample suspicion scores q(x) ∈ [0,1] by measuring how
much a sample's representation changes under targeted probe perturbations.
Backdoor-poisoned samples tend to show higher instability because the
trigger pattern is disrupted by probes while clean image semantics are preserved.

Two scoring strategies:
1. ProbeAlignmentScorer: cosine similarity between clean and probe embeddings
2. NeighborOverlapScorer: overlap of top-k neighborhoods under clean vs probe views
"""

import torch
import torch.nn.functional as F

from prcl.defenses.prcl.probes import get_probe_transforms
from prcl.defenses.prcl.thresholds import get_normalizer


class ProbeAlignmentScorer:
    """Suspicion scoring via probe-alignment instability.

    For each sample, computes cosine similarity between the clean embedding
    and each probe embedding. Lower similarity → higher suspicion.

    q_raw(x) = 1 - mean_k(cos_sim(h_clean, h_probe_k))
    q(x) = normalize(q_raw(x)) ∈ [0, 1]
    """

    def __init__(self, probe_types: list[str], normalize: str = "rolling_zscore"):
        self.probes = get_probe_transforms(probe_types)
        self.normalizer = get_normalizer(normalize)

    def compute_scores(
        self,
        clean_images: torch.Tensor,
        encoder: torch.nn.Module,
        clean_representations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute suspicion scores for a batch of images.

        Args:
            clean_images: Original images (B, C, H, W) — before SSL augmentation.
            encoder: Backbone encoder (representations, not projections).
            clean_representations: Pre-computed clean embeddings (B, D). If None,
                will be computed from clean_images.

        Returns:
            Suspicion scores q(x) ∈ [0, 1] of shape (B,).
        """
        # Get clean representations if not provided
        if clean_representations is None:
            with torch.no_grad():
                clean_representations = encoder(clean_images)

        clean_h = F.normalize(clean_representations.detach(), dim=1)

        # Compute probe representations and cosine similarities
        probe_sims = []
        for probe in self.probes:
            with torch.no_grad():
                probed_images = probe(clean_images)
                probe_h = encoder(probed_images)
                probe_h = F.normalize(probe_h, dim=1)

            # Cosine similarity per sample
            sim = (clean_h * probe_h).sum(dim=1)  # (B,)
            probe_sims.append(sim)

        # Average similarity across probes
        mean_sim = torch.stack(probe_sims, dim=0).mean(dim=0)  # (B,)

        # Raw suspicion: lower similarity → higher suspicion
        raw_scores = 1.0 - mean_sim

        # Normalize to [0, 1]
        if self.normalizer is not None:
            scores = self.normalizer.normalize(raw_scores)
        else:
            scores = raw_scores.clamp(0, 1)

        return scores.detach()

    def reset(self):
        if self.normalizer is not None:
            self.normalizer.reset()


class NeighborOverlapScorer:
    """Suspicion scoring via neighborhood overlap instability.

    Compares top-k neighbor sets of clean vs probe embeddings within the batch.
    Lower overlap → higher suspicion. More expensive than ProbeAlignmentScorer
    but captures structural changes in the representation space.

    q_raw(x) = 1 - mean_k(|neighbors_clean ∩ neighbors_probe_k| / k)
    """

    def __init__(self, probe_types: list[str], neighbor_k: int = 10,
                 normalize: str = "rolling_zscore"):
        self.probes = get_probe_transforms(probe_types)
        self.k = neighbor_k
        self.normalizer = get_normalizer(normalize)

    def compute_scores(
        self,
        clean_images: torch.Tensor,
        encoder: torch.nn.Module,
        clean_representations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = clean_images.device

        if clean_representations is None:
            with torch.no_grad():
                clean_representations = encoder(clean_images)

        clean_h = F.normalize(clean_representations.detach(), dim=1)
        batch_size = clean_h.shape[0]

        # Clamp k to available batch size
        k = min(self.k, batch_size - 1)
        if k < 1:
            return torch.zeros(batch_size, device=device)

        # Clean neighborhood: top-k indices by cosine similarity
        clean_sim = torch.mm(clean_h, clean_h.t())
        clean_sim.fill_diagonal_(-float("inf"))  # exclude self
        _, clean_neighbors = clean_sim.topk(k, dim=1)  # (B, k)

        probe_overlaps = []
        for probe in self.probes:
            with torch.no_grad():
                probed_images = probe(clean_images)
                probe_h = encoder(probed_images)
                probe_h = F.normalize(probe_h, dim=1)

            probe_sim = torch.mm(probe_h, probe_h.t())
            probe_sim.fill_diagonal_(-float("inf"))
            _, probe_neighbors = probe_sim.topk(k, dim=1)  # (B, k)

            # Compute set overlap per sample
            overlap = torch.zeros(batch_size, device=device)
            for i in range(batch_size):
                clean_set = set(clean_neighbors[i].cpu().tolist())
                probe_set = set(probe_neighbors[i].cpu().tolist())
                overlap[i] = len(clean_set & probe_set) / k

            probe_overlaps.append(overlap)

        mean_overlap = torch.stack(probe_overlaps, dim=0).mean(dim=0)
        raw_scores = 1.0 - mean_overlap

        if self.normalizer is not None:
            scores = self.normalizer.normalize(raw_scores)
        else:
            scores = raw_scores.clamp(0, 1)

        return scores.detach()

    def reset(self):
        if self.normalizer is not None:
            self.normalizer.reset()


def build_pcf_scorer(cfg):
    """Build a PCF scorer from the defense config."""
    stat = cfg.defense.pcf.stat
    probe_types = list(cfg.defense.pcf.probe_types)
    normalize = cfg.defense.pcf.normalize

    if stat == "probe_alignment":
        return ProbeAlignmentScorer(probe_types=probe_types, normalize=normalize)
    elif stat == "neighbor_overlap":
        neighbor_k = cfg.defense.pcf.neighbor_k
        return NeighborOverlapScorer(
            probe_types=probe_types, neighbor_k=neighbor_k, normalize=normalize
        )
    else:
        raise ValueError(f"Unknown PCF stat: {stat}")
