from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import FeatureSpec, count_parameters


def _split_point(point: str) -> Tuple[str, Optional[int]]:
    if "[" in point and point.endswith("]"):
        base = point[: point.index("[")]
        idx = int(point[point.index("[") + 1 : -1])
        return base, idx
    return point, None


def resolve_module(root: nn.Module, path: str) -> nn.Module:
    """Resolve dotted / indexed module paths, e.g. ``layer3.1.conv2`` or ``features[10]``."""
    cur: nn.Module = root
    tokens = path.replace("]", "").split(".")
    for token in tokens:
        if not token:
            continue
        if "[" in token:
            head, tail = token.split("[", 1)
            if head:
                cur = getattr(cur, head)
            cur = cur[int(tail)]
        else:
            if token.isdigit():
                cur = cur[int(token)]  # type: ignore[index]
            else:
                cur = getattr(cur, token)
    return cur


class FeatureHook:
    def __init__(self, module: nn.Module):
        self.output = None
        self.hook = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.output = output

    def close(self):
        self.hook.remove()


@dataclass(frozen=True)
class FeatureShape:
    kind: str
    channels: int
    height: Optional[int] = None
    width: Optional[int] = None
    tokens: Optional[int] = None
    has_cls_token: bool = False

    @classmethod
    def from_tensor(cls, x: torch.Tensor) -> "FeatureShape":
        if x.dim() == 4:
            return cls("map", x.shape[1], x.shape[2], x.shape[3])
        if x.dim() == 3:
            return cls("tokens", x.shape[2], tokens=x.shape[1], has_cls_token=True)
        if x.dim() == 2:
            return cls("vector", x.shape[1])
        raise ValueError(f"Unsupported feature tensor shape: {tuple(x.shape)}")

    def to_feature_spec(self) -> FeatureSpec:
        return FeatureSpec(
            kind=self.kind,  # type: ignore[arg-type]
            channels=self.channels,
            height=self.height,
            width=self.width,
            tokens=self.tokens,
            has_cls_token=self.has_cls_token,
        )


def _square_side(n: int) -> int:
    side = int(n ** 0.5)
    if side * side != n:
        raise ValueError(f"{n} is not square.")
    return side


class FeatureRegressor(nn.Module):
    """Lightweight source->target adapter used by the FitNets baseline."""

    def __init__(self, source: FeatureShape, target: FeatureShape, bottleneck_ratio: float = 0.5):
        super().__init__()
        self.source = source
        self.target = target
        self.bottleneck_ratio = bottleneck_ratio

        s, t = source, target
        hidden = max(16, int(min(s.channels, t.channels) * bottleneck_ratio))

        if s.kind == "map" and t.kind == "map":
            self.pool = nn.Identity() if (s.height == t.height and s.width == t.width) else nn.AdaptiveAvgPool2d((t.height or s.height or 1, t.width or s.width or 1))
            self.proj = nn.Conv2d(s.channels, t.channels, kernel_size=1, bias=False)

        elif s.kind == "tokens" and t.kind == "tokens":
            self.norm = nn.LayerNorm(s.channels)
            self.proj = nn.Sequential(
                nn.Linear(s.channels, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, t.channels, bias=False),
            )

        elif s.kind == "vector" and t.kind == "vector":
            self.proj = nn.Sequential(
                nn.LayerNorm(s.channels),
                nn.Linear(s.channels, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, t.channels, bias=False),
            )

        else:
            self.proj = nn.Sequential(
                nn.Linear(s.channels, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, t.channels, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, t = self.source, self.target

        if s.kind == "map" and t.kind == "map":
            x = self.pool(x)
            return self.proj(x)

        if s.kind == "tokens" and t.kind == "tokens":
            if x.shape[1] != (t.tokens or x.shape[1]):
                patch = x[:, 1:, :] if s.has_cls_token else x
                tgt_tokens = t.tokens or patch.shape[1]
                if t.has_cls_token:
                    tgt_patch = tgt_tokens - 1
                    if tgt_patch > 0:
                        patch = F.interpolate(patch.transpose(1, 2), size=tgt_patch, mode="linear", align_corners=False).transpose(1, 2)
                    cls = x[:, :1, :] if s.has_cls_token else patch.mean(dim=1, keepdim=True)
                    x = torch.cat([cls, patch], dim=1)
                else:
                    x = F.interpolate(patch.transpose(1, 2), size=tgt_tokens, mode="linear", align_corners=False).transpose(1, 2)
            x = self.norm(x)
            return self.proj(x)

        if s.kind == "vector" and t.kind == "vector":
            return self.proj(x)

        if s.kind == "map" and t.kind == "tokens":
            patch_tokens = t.tokens or (1 if t.has_cls_token else 1)
            if t.has_cls_token:
                patch_tokens -= 1
            side = _square_side(max(patch_tokens, 1))
            x = F.adaptive_avg_pool2d(x, (side, side)).flatten(2).transpose(1, 2).contiguous()
            x = self.proj(x)
            if t.has_cls_token:
                cls = x.mean(dim=1, keepdim=True)
                x = torch.cat([cls, x], dim=1)
            return x

        if s.kind == "tokens" and t.kind == "map":
            patch = x[:, 1:, :] if s.has_cls_token else x
            side = _square_side(max((t.height or 1) * (t.width or 1), 1))
            if patch.shape[1] != side * side:
                patch = F.interpolate(patch.transpose(1, 2), size=side * side, mode="linear", align_corners=False).transpose(1, 2)
            x = self.proj(patch)
            x = x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], side, side)
            if t.height is not None and t.width is not None and (t.height != side or t.width != side):
                x = F.interpolate(x, size=(t.height, t.width), mode="bilinear", align_corners=False)
            return x

        if s.kind == "map" and t.kind == "vector":
            x = x.mean(dim=(-2, -1))
            return self.proj(x)

        if s.kind == "tokens" and t.kind == "vector":
            x = x[:, 0] if s.has_cls_token else x.mean(dim=1)
            return self.proj(x)

        if s.kind == "vector" and t.kind == "map":
            x = self.proj(x).unsqueeze(-1).unsqueeze(-1)
            h, w = t.height or 1, t.width or 1
            return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)

        if s.kind == "vector" and t.kind == "tokens":
            x = self.proj(x).unsqueeze(1)
            n = t.tokens or 1
            return x.expand(-1, n, -1)

        raise ValueError(f"Unsupported source/target pair: {s.kind} -> {t.kind}")


class FitNetsDistiller(nn.Module):
    """FitNets baseline with a direction switch via ``reverse``."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        *,
        teacher_hints: Sequence[str],
        student_hints: Sequence[str],
        reverse: bool = False,
        bottleneck_ratio: float = 0.5,
        freeze_teacher: bool = True,
    ):
        super().__init__()
        if len(teacher_hints) != len(student_hints):
            raise ValueError("teacher_hints and student_hints must have the same length.")

        self.teacher = teacher
        self.student = student
        self.teacher_hints = list(teacher_hints)
        self.student_hints = list(student_hints)
        self.reverse = reverse
        self.bottleneck_ratio = bottleneck_ratio

        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

        self.teacher_hooks = [FeatureHook(resolve_module(self.teacher, name)) for name in self.teacher_hints]
        self.student_hooks = [FeatureHook(resolve_module(self.student, name)) for name in self.student_hints]
        self.regressors = nn.ModuleList()
        self._initialized = False

    @torch.no_grad()
    def initialize(self, sample_batch: torch.Tensor) -> None:
        """Run one dummy forward pass to infer feature sizes and build regressors."""
        self.teacher.eval()
        self.student.eval()
        _ = self.teacher(sample_batch)
        _ = self.student(sample_batch)

        self.regressors = nn.ModuleList()
        for t_hook, s_hook in zip(self.teacher_hooks, self.student_hooks):
            t_feat = t_hook.output
            s_feat = s_hook.output
            if t_feat is None or s_feat is None:
                raise RuntimeError("Failed to capture FitNets features during initialization.")

            t_shape = FeatureShape.from_tensor(t_feat)
            s_shape = FeatureShape.from_tensor(s_feat)

            source = t_shape if self.reverse else s_shape
            target = s_shape if self.reverse else t_shape
            self.regressors.append(FeatureRegressor(source, target, self.bottleneck_ratio))

        self._initialized = True

    def forward(self, x: torch.Tensor):
        if not self._initialized:
            raise RuntimeError("FitNetsDistiller must be initialized with a sample batch before training.")

        with torch.no_grad():
            teacher_logits = self.teacher(x)

        student_logits = self.student(x)

        total_hint = torch.tensor(0.0, device=x.device)
        for idx, reg in enumerate(self.regressors):
            t_feat = self.teacher_hooks[idx].output
            s_feat = self.student_hooks[idx].output
            if t_feat is None or s_feat is None:
                raise RuntimeError(f"Missing feature at pair index {idx}.")

            source = t_feat if self.reverse else s_feat
            target = s_feat if self.reverse else t_feat
            adapted = reg(source)

            if adapted.shape != target.shape:
                if adapted.dim() == 4 and target.dim() == 4:
                    adapted = F.interpolate(adapted, size=target.shape[-2:], mode="bilinear", align_corners=False)
                elif adapted.dim() == 3 and target.dim() == 3 and adapted.shape[1] != target.shape[1]:
                    adapted = F.interpolate(adapted.transpose(1, 2), size=target.shape[1], mode="linear", align_corners=False).transpose(1, 2)

            total_hint = total_hint + F.mse_loss(adapted, target)

        return teacher_logits, student_logits, total_hint

    def parameter_summary(self) -> Dict[str, int]:
        return {
            "teacher_params": count_parameters(self.teacher),
            "student_params": count_parameters(self.student),
            "regressor_params": count_parameters(self.regressors),
            "trainable_teacher_params": count_parameters(self.teacher, trainable_only=True),
            "trainable_student_params": count_parameters(self.student, trainable_only=True),
            "trainable_regressor_params": count_parameters(self.regressors, trainable_only=True),
        }

    def close_hooks(self) -> None:
        for hook in self.teacher_hooks + self.student_hooks:
            hook.close()
