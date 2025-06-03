import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Label Smoothing Loss with Padding Masking
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, label_smoothing=0.1, vocab_size=10000, ignore_index=0):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        pred: (batch_size, seq_len, vocab_size)
        target: (batch_size, seq_len)
        """
        pred = pred.view(-1, self.vocab_size)
        target = target.view(-1)

        # Build smoothed label distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))

            # Ignore index (padding)
            ignore_mask = (target == self.ignore_index)
            target_clamped = target.clone()
            target_clamped[ignore_mask] = 0  # Prevent indexing error

            true_dist.scatter_(1, target_clamped.unsqueeze(1), 1.0 - self.smoothing)
            true_dist[ignore_mask] = 0  # Zero-out ignored labels

        # KL-divergence loss with masking
        loss = torch.sum(-true_dist * F.log_softmax(pred, dim=-1), dim=-1)
        loss = loss.masked_fill(ignore_mask, 0)

        return loss.sum() / (~ignore_mask).sum().clamp(min=1)


# Optimizer with AdamW
def get_optimizer(model, lr=3e-4, weight_decay=0.01):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
