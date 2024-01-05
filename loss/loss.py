import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.Module):
    def __init__(self, weights, margin):
        super(MarginLoss, self).__init__()
        self.class_weights = weights
        self.margin = margin

    def forward(self, inputs, targets, target_oe, test=False):
        if self.class_weights is not None:
            class_weights = torch.FloatTensor(self.class_weights).to(inputs.device)
        else:
            class_weights = None
        loss = F.cross_entropy(inputs[: len(targets)], targets, weight=class_weights)

        all_targets = torch.cat((targets, target_oe), 0)
        if not test:
            ood_mask = all_targets == target_oe[0]
            id_mask = all_targets != target_oe[0]

            ood_score = inputs[ood_mask]
            # ood_score = inputs[ood_mask][:, target_oe]
            best_id_score, _ = torch.max(inputs[id_mask], dim=1)
            best_id_score_mean = torch.mean(ood_score)
            margin_loss = torch.mean(
                torch.clamp(self.margin - (best_id_score_mean - best_id_score), min=0.0)
            )
            final_loss = margin_loss + loss
            return final_loss
        else:
            return loss
