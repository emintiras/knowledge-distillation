import torch
import torch.nn as nn

def distillation_loss(student_logits, teacher_logits, labels, T=4, alpha=0.6):
    soft_teacher = torch.softmax(teacher_logits / T, dim=1)
    soft_student = torch.log_softmax(student_logits / T, dim=1)
    kl_loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (T ** 2)
    ce_loss = nn.CrossEntropyLoss()(student_logits, labels)
    return alpha * ce_loss + (1 - alpha) * kl_loss