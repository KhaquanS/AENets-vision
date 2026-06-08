from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import FeatureSpec, describe_feature_spec


@dataclass(frozen=True)
class AEOutput:
    reconstruction: torch.Tensor
    latent: torch.Tensor
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    sparsity_loss: torch.Tensor


def _square_side(n: int) -> int:
    side = int(n ** 0.5)
    if side * side != n:
        raise ValueError(f"{n} is not a square number.")
    return side


class LightweightFeatureAutoencoder(nn.Module):
    """
    Small, modality-aware translator used by SAE injection.

    The returned latent is the student-facing representation.
    """

    def __init__(
        self,
        source_spec: FeatureSpec,
        target_spec: FeatureSpec,
        *,
        bottleneck_ratio: float = 0.5,
        sparsity_weight: float = 1e-4,
    ):
        super().__init__()
        self.source_spec = source_spec
        self.target_spec = target_spec
        self.sparsity_weight = sparsity_weight
        self._cached_source_shape: Optional[torch.Size] = None

        s, t = source_spec, target_spec
        self.kind_pair = (s.kind, t.kind)

        if s.kind == t.kind == "map":
            bottleneck = max(16, int(min(s.channels, t.channels) * bottleneck_ratio))
            self.encoder = nn.Sequential(
                nn.Conv2d(s.channels, bottleneck, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(bottleneck, t.channels, kernel_size=1, bias=False),
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(t.channels, bottleneck, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(bottleneck, s.channels, kernel_size=1, bias=False),
            )

        elif s.kind == t.kind == "tokens":
            bottleneck = max(32, int(min(s.channels, t.channels) * bottleneck_ratio))
            self.encoder = nn.Sequential(
                nn.LayerNorm(s.channels),
                nn.Linear(s.channels, bottleneck, bias=False),
                nn.GELU(),
                nn.Linear(bottleneck, t.channels, bias=False),
            )
            self.decoder = nn.Sequential(
                nn.LayerNorm(t.channels),
                nn.Linear(t.channels, bottleneck, bias=False),
                nn.GELU(),
                nn.Linear(bottleneck, s.channels, bias=False),
            )

        elif s.kind == t.kind == "vector":
            bottleneck = max(32, int(min(s.channels, t.channels) * bottleneck_ratio))
            self.encoder = nn.Sequential(
                nn.LayerNorm(s.channels),
                nn.Linear(s.channels, bottleneck, bias=False),
                nn.GELU(),
                nn.Linear(bottleneck, t.channels, bias=False),
            )
            self.decoder = nn.Sequential(
                nn.LayerNorm(t.channels),
                nn.Linear(t.channels, bottleneck, bias=False),
                nn.GELU(),
                nn.Linear(bottleneck, s.channels, bias=False),
            )

        elif s.kind == "map" and t.kind == "tokens":
            bottleneck = max(32, int(min(s.channels, t.channels) * bottleneck_ratio))
            self.encoder = nn.Sequential(
                nn.LayerNorm(s.channels),
                nn.Linear(s.channels, bottleneck, bias=False),
                nn.GELU(),
                nn.Linear(bottleneck, t.channels, bias=False),
            )
            self.decoder = nn.Sequential(
                nn.LayerNorm(t.channels),
                nn.Linear(t.channels, bottleneck, bias=False),
                nn.GELU(),
                nn.Linear(bottleneck, s.channels, bias=False),
            )

        elif s.kind == "tokens" and t.kind == "map":
            bottleneck = max(32, int(min(s.channels, t.channels) * bottleneck_ratio))
            self.encoder = nn.Sequential(
                nn.LayerNorm(s.channels),
                nn.Linear(s.channels, bottleneck, bias=False),
                nn.GELU(),
                nn.Linear(bottleneck, t.channels, bias=False),
            )
            self.decoder = nn.Sequential(
                nn.LayerNorm(t.channels),
                nn.Linear(t.channels, bottleneck, bias=False),
                nn.GELU(),
                nn.Linear(bottleneck, s.channels, bias=False),
            )

        else:
            bottleneck = max(32, int(min(s.channels, t.channels) * bottleneck_ratio))
            self.encoder = nn.Sequential(
                nn.LayerNorm(s.channels) if s.kind == "tokens" else nn.Identity(),
                nn.Linear(s.channels, bottleneck, bias=False),
                nn.GELU(),
                nn.Linear(bottleneck, t.channels, bias=False),
            )
            self.decoder = nn.Sequential(
                nn.LayerNorm(t.channels) if t.kind == "tokens" else nn.Identity(),
                nn.Linear(t.channels, bottleneck, bias=False),
                nn.GELU(),
                nn.Linear(bottleneck, s.channels, bias=False),
            )

    def _cache(self, x: torch.Tensor):
        self._cached_source_shape = x.shape

    def _source_hw(self) -> Tuple[int, int]:
        if self.source_spec.height is not None and self.source_spec.width is not None:
            return self.source_spec.height, self.source_spec.width
        if self._cached_source_shape is not None and len(self._cached_source_shape) == 4:
            return int(self._cached_source_shape[-2]), int(self._cached_source_shape[-1])
        if self.source_spec.tokens is not None:
            n = self.source_spec.tokens - (1 if self.source_spec.has_cls_token else 0)
            side = _square_side(n)
            return side, side
        return 7, 7

    def _target_hw(self) -> Tuple[int, int]:
        if self.target_spec.height is not None and self.target_spec.width is not None:
            return self.target_spec.height, self.target_spec.width
        if self.target_spec.tokens is not None:
            n = self.target_spec.tokens - (1 if self.target_spec.has_cls_token else 0)
            side = _square_side(n)
            return side, side
        return 7, 7

    def _map_to_tokens(self, x: torch.Tensor, spec: FeatureSpec) -> torch.Tensor:
        h, w = self._target_hw() if spec is self.target_spec else self._source_hw()
        x = F.adaptive_avg_pool2d(x, (h, w))
        x = x.flatten(2).transpose(1, 2).contiguous()
        if spec.has_cls_token:
            cls = x.mean(dim=1, keepdim=True)
            x = torch.cat([cls, x], dim=1)
        return x

    def _tokens_to_map(self, x: torch.Tensor, spec: FeatureSpec) -> torch.Tensor:
        tokens = x[:, 1:, :] if spec.has_cls_token and x.shape[1] > 1 else x
        h, w = self._target_hw() if spec is self.target_spec else self._source_hw()

        # Prefer the patch-token count. If that is not square, try dropping a CLS token.
        if tokens.shape[1] != h * w:
            candidate = None
            for m in (tokens.shape[1], tokens.shape[1] - 1 if tokens.shape[1] > 1 else tokens.shape[1]):
                if m > 0:
                    side = int(m ** 0.5)
                    if side * side == m:
                        candidate = side
                        if m != tokens.shape[1]:
                            tokens = tokens[:, 1:, :]
                        break
            if candidate is None:
                side = int(tokens.shape[1] ** 0.5)
                if side * side != tokens.shape[1]:
                    side = int(max((tokens.shape[1] - 1), 1) ** 0.5)
                candidate = max(side, 1)
            h = w = candidate
        return tokens.transpose(1, 2).contiguous().view(tokens.shape[0], tokens.shape[2], h, w)

    def _vector_to_map(self, x: torch.Tensor, spec: FeatureSpec) -> torch.Tensor:
        h, w = self._target_hw() if spec is self.target_spec else self._source_hw()
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

    def _map_to_vector(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(-2, -1))

    def _tokens_to_vector(self, x: torch.Tensor, spec: FeatureSpec) -> torch.Tensor:
        return x[:, 0] if spec.has_cls_token else x.mean(dim=1)

    def _vector_to_tokens(self, x: torch.Tensor, spec: FeatureSpec) -> torch.Tensor:
        n = spec.tokens or 1
        return x.unsqueeze(1).expand(-1, n, -1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        s, t = self.source_spec, self.target_spec
        self._cache(x)

        if s.kind == "map":
            if t.kind == "map":
                return self.encoder(x)
            if t.kind == "tokens":
                tokens = self._map_to_tokens(x, t)
                return self.encoder(tokens)
            return self.encoder(self._map_to_vector(x))

        if s.kind == "tokens":
            if t.kind == "tokens":
                return self.encoder(x)
            if t.kind == "map":
                tokens = self.encoder(x)
                return self._tokens_to_map(tokens, t)
            return self.encoder(self._tokens_to_vector(x, s))

        if s.kind == "vector":
            if x.dim() != 2:
                x = x.flatten(1)
            if t.kind == "vector":
                return self.encoder(x)
            if t.kind == "map":
                return self._vector_to_map(self.encoder(x), t)
            if t.kind == "tokens":
                return self._vector_to_tokens(self.encoder(x), t)

        raise ValueError(f"Unsupported source/target pair: {describe_feature_spec(s)} -> {describe_feature_spec(t)}")

    def _decode_to_source(self, latent: torch.Tensor) -> torch.Tensor:
        s, t = self.source_spec, self.target_spec

        if s.kind == "map":
            if t.kind == "map":
                recon = self.decoder(latent)
            elif t.kind == "tokens":
                if latent.dim() != 3:
                    raise ValueError("Expected token latent for map<-tokens reconstruction.")
                recon = self.decoder(latent)
                recon = self._tokens_to_map(recon, s)
            else:  # vector target
                if latent.dim() != 2:
                    latent = latent.flatten(1)
                recon = self.decoder(latent)
                recon = self._vector_to_map(recon, s)
            return recon

        if s.kind == "tokens":
            if t.kind == "tokens":
                recon = self.decoder(latent)
            elif t.kind == "map":
                if latent.dim() != 4:
                    raise ValueError("Expected map latent for tokens<-map reconstruction.")
                recon = self._map_to_tokens(latent, s)
                recon = self.decoder(recon)
            else:  # vector target
                if latent.dim() != 2:
                    latent = latent.flatten(1)
                recon = self.decoder(latent)
                recon = self._vector_to_tokens(recon, s)
            return recon

        if s.kind == "vector":
            if t.kind == "vector":
                recon = self.decoder(latent)
            elif t.kind == "map":
                if latent.dim() != 4:
                    latent = latent.flatten(1)
                recon = self._map_to_vector(latent)
                recon = self.decoder(recon)
            else:  # tokens target
                if latent.dim() != 3:
                    latent = latent.flatten(1)
                recon = self._tokens_to_vector(latent, t)
                recon = self.decoder(recon)
            return recon

        raise ValueError("Unsupported source kind.")

    def forward(self, x: torch.Tensor):
        latent = self._encode(x)
        reconstruction = self._decode_to_source(latent)

        recon_target = x
        if reconstruction.dim() != x.dim():
            recon_target = x.flatten(1)
            reconstruction = reconstruction.flatten(1)

        reconstruction_loss = F.mse_loss(reconstruction, recon_target)
        sparsity_loss = self.sparsity_weight * latent.abs().mean()
        total_loss = reconstruction_loss + sparsity_loss
        return reconstruction, latent, total_loss, reconstruction_loss, sparsity_loss


class SparseAutoencoder(LightweightFeatureAutoencoder):
    def __init__(self, teacher_channels: int, student_channels: int, lambda_sparsity: float = 1e-4):
        super().__init__(
            FeatureSpec(kind="map", channels=teacher_channels),
            FeatureSpec(kind="map", channels=student_channels),
            sparsity_weight=lambda_sparsity,
        )
