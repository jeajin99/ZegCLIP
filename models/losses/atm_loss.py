import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES
from .criterion import SegPlusCriterion

@LOSSES.register_module()
class SegLossPlus(nn.Module):
    """ATMLoss.
    """
    def __init__(self,
                 num_classes,
                 dec_layers,
                 mask_weight=20.0,
                 tversky_weight=1.0,  # Tversky 손실로 이름 변경
                 loss_weight=1.0,
                 use_point=False,
                 alpha=0.7,  # Tversky 손실을 위한 추가 파라미터
                 beta=0.7):
        super(SegLossPlus, self).__init__()
        
        # Tversky 손실을 반영한 가중치 딕셔너리 업데이트
        weight_dict = {"loss_mask": mask_weight, "loss_tversky": tversky_weight}
        
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        # SegPlusCriterion에 Tversky 손실의 alpha와 beta 파라미터 전달
        self.criterion = SegPlusCriterion(
            num_classes,
            weight_dict=weight_dict,
            losses=["masks"],
            alpha=alpha,
            beta=beta
        )

        self.loss_weight = loss_weight

    def forward(self,
                outputs,
                label,
                ignore_index=255,
                ):
        """Forward 함수."""
        
        self.ignore_index = ignore_index
        targets = self.prepare_targets(label)
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] = losses[k] * self.criterion.weight_dict[k] * self.loss_weight
            else:
                # `weight_dict`에 명시되지 않은 손실은 제거
                losses.pop(k)

        return losses

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            # gt_cls
            gt_cls = targets_per_image.unique()
            gt_cls = gt_cls[gt_cls != self.ignore_index]
            masks = []
            for cls in gt_cls:
                masks.append(targets_per_image == cls)
            if len(gt_cls) == 0:
                masks.append(targets_per_image == self.ignore_index)

            masks = torch.stack(masks, dim=0)
            new_targets.append(
                {
                    "labels": gt_cls,
                    "target_masks": masks,
                    "masks": targets_per_image,
                }
            )
        return new_targets
