import abc
import torch
import monai
from typing import Dict, Optional


class TrainingCriterion(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    def forward(self, prediction_logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute the training criterion.

        Arguments
        ---------
        prediction_logits: torch.Tensor
            Logits for model prediction in a segmentation task.
        targets: torch.Tensor
            Ground-truth segmentation mask labels.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary mapping strings to the loss values. The string ``"loss"`` must map to the overall loss term, but
            additional keys can contain more information that will be logged with the output.
        """
        raise NotImplementedError("This is an abstract class!")


class BCECriterion(TrainingCriterion):
    """
    Standard binary cross entropy training loss.
    """
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, prediction_logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss = self.loss(prediction_logits, targets)
        return {"loss": loss, "BCE_loss": loss}
    

class MSECriterion(TrainingCriterion):
    def __init__(self):
        super().__init__()
        self.crit = torch.nn.MSELoss(reduction='sum')

    def forward(self, prediction_logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred = torch.sigmoid(prediction_logits)
        loss = self.crit(pred, targets)
        return {"loss": loss, "MSE_loss": loss}


class DiceCriterion(TrainingCriterion):
    """
    Dice loss term. Dice loss considers the agreement (2 x intersection / union) of predicted and true segmentation
    masks.

    Arguments
    ---------
    include_background: bool = True
        False if channel index 0 should be ignored from all calculations.
    use_sigmoid: bool = True
        True iff we should apply sigmoid activation to the predicted values (for Dice loss only)
    smooth: float = 1e-6
        Value to add to both numerator and denominator for Dice numerical stability
    """
    def __init__(self, include_background: bool = True, use_sigmoid: bool = True, smooth: float = 1e-6,):
        super().__init__()
        self.loss = monai.losses.DiceLoss(include_background=include_background, sigmoid=use_sigmoid, smooth_nr=smooth,
                                          smooth_dr=smooth)

    def forward(self, prediction_logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss = self.loss(prediction_logits, targets)
        return {"loss": loss, "dice_loss": loss}


class FocalCriterion(TrainingCriterion):
    """
    Focal loss term. Focal loss is a variant of BCE that can emphasize hard-to-classify samples with a ``gamma > 0``.

    Arguments
    ---------
    gamma: float = 2.0
        Gamma exponent term for the focal loss. Larger terms place more emphasis on misclassified samples.
    include_background: bool = True
        False if channel index 0 should be ignored from all calculations.
    alpha: Optional[float] = None
        Only for alpha-balanced focal loss; gives background class weight ``(1-alpha)`` and weight ``alpha`` for all
        other classes.
    """
    def __init__(self, gamma: float=2.0, include_background: bool = True, alpha: Optional[float] = None):
        super().__init__()
        self.loss = monai.losses.FocalLoss(include_background=include_background, gamma=gamma, alpha=alpha)

    def forward(self, prediction_logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss = self.loss(prediction_logits, targets)
        return {"loss": loss, "focal_loss": loss}


class ComboDiceBCECriterion(TrainingCriterion):
    """
    Training loss term that combines both BCE loss and Dice loss. The BCE loss considers the probability that the
    segmentation was correct for each pixel; Dice loss considers the agreement (2 x intersection / union) of predicted
    and true segmentation masks.

    Arguments
    ---------
    include_background: bool = True
        False if channel index 0 should be ignored from all calculations.
    use_sigmoid: bool = True
        True iff we should apply sigmoid activation to the predicted values (for Dice loss only)
    dice_smooth: float = 1e-6
        Value to add to both numerator and denominator for Dice numerical stability
    dice_weight: float = 0.5
    """
    def __init__(self, include_background: bool = True, use_sigmoid: bool = True, dice_smooth: float = 1e-6,
                 dice_weight: float = 0.5):
        super().__init__()
        self.loss = monai.losses.DiceCELoss(
            include_background=include_background, sigmoid=use_sigmoid, smooth_nr=dice_smooth, smooth_dr=dice_smooth,
            lambda_dice=dice_weight, lambda_ce=(1-dice_weight))
        self.dice_weight = dice_weight

    def forward(self, prediction_logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        bce_loss = self.loss.bce(prediction_logits, targets)
        dice_loss = self.loss.dice(prediction_logits, targets)
        loss = (1 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss
        return {"loss": loss, "bce_loss": bce_loss, "dice_loss": dice_loss}


class ComboDiceFocalCriterion(TrainingCriterion):
    """
    Training loss term that combines both Dice loss and Focal loss. The Dice loss considers the agreement
    (2 x intersection / union) of predicted and true segmentation masks; Focal loss is a variant of BCE that can
    emphasize hard-to-classify samples with a ``focal_gamma > 0``.

    Arguments
    ---------
    include_background: bool = True
        False if channel index 0 should be ignored from all calculations.
    use_sigmoid: bool = True
        True iff we should apply sigmoid activation to the predicted values (for Dice loss only)
    dice_smooth: float = 1e-6
        Value to add to both numerator and denominator for Dice numerical stability
    dice_weight: float = 0.5
        Weight for the Dice component of the overall loss term (``1-dice_weight`` will be the focal loss contribution).
    focal_gamma: float = 2.0
        Gamma exponent term for the focal loss.
    """
    def __init__(self, include_background: bool = True, use_sigmoid: bool = True, dice_smooth: float = 1e-6,
                 dice_weight: float = 0.5, focal_gamma: float = 2.0):
        super().__init__()
        self.loss = monai.losses.DiceFocalLoss(
            include_background=include_background, sigmoid=use_sigmoid, smooth_nr=dice_smooth, smooth_dr=dice_smooth,
            gamma=focal_gamma, lambda_dice=dice_weight, lambda_focal=1-dice_weight)
        self.dice_weight = dice_weight

    def forward(self, prediction_logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        dice_loss = self.loss.dice(prediction_logits, targets)
        focal_loss = self.loss.focal(prediction_logits, targets)
        loss = (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss
        return {"loss": loss, "focal_loss": focal_loss, "dice_loss": dice_loss}