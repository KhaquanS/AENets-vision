from __future__ import annotations

import torch
import torch.nn as nn


class LogitKD(nn.Module):
    """Vanilla Hinton-style KD wrapper.

    The module only couples teacher and student forward passes. Losses are
    computed in the training loop so the same wrapper stays compatible with the
    repo's common training API.
    """

    def __init__(self, teacher: nn.Module, student: nn.Module, freeze_teacher: bool = True):
        super().__init__()
        self.teacher = teacher
        self.student = student

        if freeze_teacher:
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.teacher.eval()

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)
        return teacher_logits, student_logits
