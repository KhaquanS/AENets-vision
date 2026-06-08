from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: float,
    temperature: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Decoupled KD from https://arxiv.org/abs/2203.08679."""
    target_mask = torch.zeros_like(student_logits).scatter_(1, targets.unsqueeze(1), 1.0)

    s_prob = F.softmax(student_logits / temperature, dim=1)
    t_prob = F.softmax(teacher_logits / temperature, dim=1)

    # target-class KD
    s_t = (s_prob * target_mask).sum(dim=1, keepdim=True)
    t_t = (t_prob * target_mask).sum(dim=1, keepdim=True)
    s_bin = torch.cat([s_t, 1.0 - s_t], dim=1).clamp_min(eps)
    t_bin = torch.cat([t_t, 1.0 - t_t], dim=1).clamp_min(eps)
    tckd = F.kl_div(torch.log(s_bin), t_bin, reduction="batchmean") * (temperature ** 2)

    # non-target KD
    s_nt = F.softmax((student_logits - 1e9 * target_mask) / temperature, dim=1)
    t_nt = F.softmax((teacher_logits - 1e9 * target_mask) / temperature, dim=1)
    nckd = F.kl_div(torch.log(s_nt.clamp_min(eps)), t_nt, reduction="batchmean") * (temperature ** 2)

    return alpha * tckd + beta * nckd


class DKD(nn.Module):
    """Teacher/student wrapper for DKD training."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        temperature: float = 4.0,
        freeze_teacher: bool = True,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def forward(self, x: torch.Tensor, targets: torch.Tensor):
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)
        loss = dkd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            targets=targets,
            alpha=self.alpha,
            beta=self.beta,
            temperature=self.temperature,
        )
        return teacher_logits, student_logits, loss
