from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from autoencoders import LightweightFeatureAutoencoder
from utils import (
    BackboneSlice,
    FeatureSpec,
    build_torchvision_model,
    count_parameters,
    freeze_module,
    infer_backbone_family,
    teacher_prefix_slice,
    student_suffix_slice,
)


@dataclass(frozen=True)
class SAEInjectionOutput:
    logits: torch.Tensor
    sae_loss: torch.Tensor
    reconstruction: torch.Tensor
    latent: torch.Tensor


class SAEInjection(nn.Module):
    """
    Teacher prefix -> lightweight autoencoder -> student suffix.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        *,
        teacher_cut: Optional[str] = None,
        student_start: Optional[str] = None,
        source_spec: Optional[FeatureSpec] = None,
        target_spec: Optional[FeatureSpec] = None,
        adapter_bottleneck_ratio: float = 0.5,
        adapter_sparsity: float = 1e-4,
        freeze_teacher: bool = True,
        freeze_student: bool = False,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.freeze_teacher = freeze_teacher
        self.freeze_student = freeze_student

        self.teacher_family = infer_backbone_family(teacher)
        self.student_family = infer_backbone_family(student)

        self.teacher_prefix = teacher_prefix_slice(teacher, teacher_cut)
        self.student_suffix = student_suffix_slice(student, student_start)

        source_spec = source_spec or self.teacher_prefix.output_spec
        target_spec = target_spec or self.student_suffix.input_spec
        if source_spec is None or target_spec is None:
            raise ValueError("Could not infer adapter source/target specs.")

        self.adapter = LightweightFeatureAutoencoder(
            source_spec=source_spec,
            target_spec=target_spec,
            bottleneck_ratio=adapter_bottleneck_ratio,
            sparsity_weight=adapter_sparsity,
        )

        if freeze_teacher:
            freeze_module(self.teacher)
        if freeze_student:
            freeze_module(self.student)

    @classmethod
    def from_torchvision_names(
        cls,
        teacher_name: str,
        student_name: str,
        *,
        num_classes: int,
        teacher_weights: Optional[object] = None,
        student_weights: Optional[object] = None,
        teacher_cut: Optional[str] = None,
        student_start: Optional[str] = None,
        freeze_teacher: bool = True,
        freeze_student: bool = False,
        adapter_bottleneck_ratio: float = 0.5,
        adapter_sparsity: float = 1e-4,
    ) -> "SAEInjection":
        teacher = build_torchvision_model(teacher_name, num_classes=num_classes, weights=teacher_weights, pretrained=teacher_weights is not None)
        student = build_torchvision_model(student_name, num_classes=num_classes, weights=student_weights, pretrained=student_weights is not None)
        return cls(
            teacher,
            student,
            teacher_cut=teacher_cut,
            student_start=student_start,
            freeze_teacher=freeze_teacher,
            freeze_student=freeze_student,
            adapter_bottleneck_ratio=adapter_bottleneck_ratio,
            adapter_sparsity=adapter_sparsity,
        )

    def forward(self, x: torch.Tensor, *, return_aux: bool = True):
        with torch.set_grad_enabled(not self.freeze_teacher):
            teacher_features = self.teacher_prefix.module(x)

        reconstruction, latent, sae_loss, recon_loss, sparsity_loss = self.adapter(teacher_features)
        logits = self.student_suffix.module(latent)

        if return_aux:
            return SAEInjectionOutput(logits=logits, sae_loss=sae_loss, reconstruction=reconstruction, latent=latent)
        return logits, sae_loss

    def parameter_summary(self) -> dict:
        return {
            "teacher_params": count_parameters(self.teacher),
            "student_params": count_parameters(self.student),
            "adapter_params": count_parameters(self.adapter),
            "trainable_teacher_params": count_parameters(self.teacher, trainable_only=True),
            "trainable_student_params": count_parameters(self.student, trainable_only=True),
            "trainable_adapter_params": count_parameters(self.adapter, trainable_only=True),
        }


def build_sae_injection(
    teacher_name: str,
    student_name: str,
    *,
    num_classes: int,
    teacher_weights: Optional[object] = None,
    student_weights: Optional[object] = None,
    teacher_cut: Optional[str] = None,
    student_start: Optional[str] = None,
    freeze_teacher: bool = True,
    freeze_student: bool = False,
    adapter_bottleneck_ratio: float = 0.5,
    adapter_sparsity: float = 1e-4,
) -> SAEInjection:
    return SAEInjection.from_torchvision_names(
        teacher_name,
        student_name,
        num_classes=num_classes,
        teacher_weights=teacher_weights,
        student_weights=student_weights,
        teacher_cut=teacher_cut,
        student_start=student_start,
        freeze_teacher=freeze_teacher,
        freeze_student=freeze_student,
        adapter_bottleneck_ratio=adapter_bottleneck_ratio,
        adapter_sparsity=adapter_sparsity,
    )
