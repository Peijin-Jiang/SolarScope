import numpy as np

def segmentation_metrics(masks, pred):
    """
    Calculate common segmentation metrics: Dice coefficient, IoU, Pixel Accuracy, Precision, and Recall.

    Parameters:
    masks (numpy.ndarray): Ground truth masks, shape (N, W, H).
    pred (numpy.ndarray): Predicted masks, shape (N, W, H).

    Returns:
    dict: A dictionary with Dice, IoU, Pixel Accuracy, Precision, and Recall for each mask.
    """

    def dice_coefficient(mask, pred):
        """Calculate the Dice coefficient."""
        intersection = np.sum(mask * pred)
        union = np.sum(mask) + np.sum(pred)
        dice = (2 * intersection) / (union + 1e-6)  # Add small epsilon to avoid division by zero
        return dice

    def iou(mask, pred):
        """Calculate the Intersection over Union (IoU)."""
        intersection = np.sum(mask * pred)
        union = np.sum(mask) + np.sum(pred) - intersection
        iou_score = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
        return iou_score

    def pixel_accuracy(mask, pred):
        """Calculate the pixel accuracy."""
        correct = np.sum(mask == pred)
        total = mask.size
        accuracy = correct / total
        return accuracy

    def precision(mask, pred):
        """Calculate the precision."""
        TP = np.sum((mask == 1) & (pred == 1))
        FP = np.sum((mask == 0) & (pred == 1))
        precision_score = TP / (TP + FP + 1e-6)  # Add small epsilon to avoid division by zero
        return precision_score

    def recall(mask, pred):
        """Calculate the recall."""
        TP = np.sum((mask == 1) & (pred == 1))
        FN = np.sum((mask == 1) & (pred == 0))
        recall_score = TP / (TP + FN + 1e-6)  # Add small epsilon to avoid division by zero
        return recall_score

    metrics = {'dice': [], 'iou': [], 'pixel_accuracy': [], 'precision': [], 'recall': []}

    for i in range(masks.shape[0]):
        mask = masks[i]
        pred_mask = pred[i]
        metrics['dice'].append(dice_coefficient(mask, pred_mask))
        metrics['iou'].append(iou(mask, pred_mask))
        metrics['pixel_accuracy'].append(pixel_accuracy(mask, pred_mask))
        metrics['precision'].append(precision(mask, pred_mask))
        metrics['recall'].append(recall(mask, pred_mask))

    # Compute the mean of each metric
    metrics = {k: np.mean(v) for k, v in metrics.items()}
    return metrics


# Example usage:
# masks = np.array([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])  # Example masks
# pred = np.array([[[0.9, 0.1], [0.2, 0.8]], [[0.7, 0.6], [0.1, 0.4]]])  # Example predictions
# print(segmentation_metrics(masks, pred))
