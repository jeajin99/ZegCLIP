import torch
import torch.nn.functional as F
from torch import nn
from .misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

import torch.distributed as dist

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def Tversky_loss(inputs, targets, num_masks, alpha=0.5, beta=0.5):
    """
    Compute the Tversky loss, which generalizes the Dice loss to provide more control over 
    the false positive and false negative penalties.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    true_pos = (inputs * targets).sum(-1)
    false_neg = ((1 - inputs) * targets).sum(-1)
    false_pos = (inputs * (1 - targets)).sum(-1)
    tversky_index = (true_pos + 1) / (true_pos + alpha * false_pos + beta * false_neg + 1)
    loss = 1 - tversky_index
    return loss.sum() / num_masks

def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection.
    """
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    prob = inputs.sigmoid()
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_masks

def cosine_margin_loss(q, e, labels, tau=1.0, m=0.5):
    assert q.shape[1]+1 == e.shape[0]
    bs, n_cls, n_dim = q.shape
    q = q.reshape(bs*n_cls, n_dim)
    pos = torch.exp(F.cosine_similarity(q, e[labels.long()].reshape(bs*n_cls, n_dim)) / tau)
    neg = torch.exp(F.cosine_similarity(q.unsqueeze(1), e.unsqueeze(0), dim=-1) / tau)
    neg = torch.sum(neg, dim=-1) + m
    return 1 - torch.mean(torch.div(pos, neg))


class SegPlusCriterion(nn.Module):
    """This class computes the loss for DETR."""

    def __init__(self, num_classes, weight_dict, losses, eos_coef=0.1, alpha=0.5, beta=0.5):
        """
        Create the criterion.
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.alpha = alpha
        self.beta = beta
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: focal loss and Tversky loss."""
        assert "pred_masks" in outputs

        # Focal loss
        src_masks = outputs["pred_masks"]
        target_masks = self._get_target_mask_binary_cross_entropy(src_masks, targets)

        bs, n_cls, H, W = target_masks.size()
        _, _, H_, W_ = src_masks.size()
        src_masks = src_masks.reshape(bs * n_cls, H_, W_)
        target_masks = target_masks.reshape(bs * n_cls, H, W)
        
        src_masks = F.interpolate(
            src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1)

        # Tversky loss
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks_tversky = outputs["pred_masks"]
        
        if src_masks_tversky.dim() != 4:
            return {"no_loss": 0}
        
        src_masks_tversky = src_masks_tversky[src_idx]
        masks_tversky = [t["target_masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks_tversky, valid = nested_tensor_from_tensor_list(masks_tversky).decompose()
        target_masks_tversky = target_masks_tversky.to(src_masks_tversky)
        target_masks_tversky = target_masks_tversky[tgt_idx]

        # upsample predictions to the target size --> for aug_loss
        src_masks_tversky = F.interpolate(
            src_masks_tversky[:, None], size=target_masks_tversky.shape[-2:], mode="bilinear", align_corners=False
        )
        src_masks_tversky = src_masks_tversky[:, 0].flatten(1)
        target_masks_tversky = target_masks_tversky.flatten(1).view(src_masks_tversky.shape)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_tversky": Tversky_loss(src_masks_tversky, target_masks_tversky, num_masks, self.alpha, self.beta),
        }
        return losses

    def _get_target_mask_binary_cross_entropy(self, out_masks, targets):
        B, C = out_masks.size()[:2]
        H, W = targets[0]['masks'].size()
        target_masks_o = torch.zeros(B, C, H * W).to(out_masks.device)
        for i, target in enumerate(targets):
            mask = target['masks'].long().reshape(-1)
            idx = torch.arange(0, H * W, 1).long().to(out_masks.device)
            mask_o = mask[mask != 255]
            idx = idx[mask != 255]
            target_masks_o[i, mask_o, idx] = 1
        return target_masks_o.reshape(B, C, H, W)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {"masks": self.loss_masks}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        labels = [x['labels'] for x in targets]
        indices_new = [[label, torch.arange(len(label))] for label in labels]
        indices = indices_new
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                aux_loss = ["masks"]
                for loss in aux_loss:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
