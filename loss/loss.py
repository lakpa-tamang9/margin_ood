import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginLoss(nn.Module):
    def __init__(self, class_weights, margin):
        super(MarginLoss, self).__init__()
        self.class_weights = class_weights
        self.margin = margin
        
    def forward(self, inputs, targets, ood_class_idx):
        if self.class_weights is not None:
            class_weights = torch.FloatTensor(self.class_weights).to(inputs.device)
        else:
            class_weights = None
        loss = F.cross_entropy(inputs, targets, weight = class_weights)
        ood_mask = (targets == ood_class_idx)
        id_mask = (targets != ood_class_idx)
        
        ood_score = inputs[ood_mask][:,ood_class_idx]
        best_id_score, _ = torch.max(inputs[id_mask][:, :ood_class_idx], dim=1)
        margin_loss = torch.mean(torch.clamp(self.margin - (best_id_score - ood_score), min = 0.0))
        final_loss = margin_loss + loss
        return final_loss