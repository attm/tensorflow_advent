import numpy as np
import matplotlib.pyplot as plt


def self_info_to_entropy_map(si_ndarray : np.ndarray) -> np.ndarray:
    """Converts self-information tensor to grayscale entropy map.

    Args:
        si_ndarray (np.ndarray): self-information tensor (processed probabilities output of segmentation model).
    Returns:
        entropy_map (np.ndarray): single-channel image representing entropy of prediction.
    """
    si_sum = np.sum(si_ndarray, axis=-1)
    si_sum *= 1.0 / si_sum.max()
    si_sum = np.expand_dims(si_sum, axis=-1)
    return si_sum

def make_model_output_subplot(x_target : np.ndarray, x_source : np.ndarray, y : np.ndarray, target_prediction : np.ndarray, source_prediction : np.ndarray, entropy_map : np.ndarray) -> np.ndarray:
    """Creates subplot from several given images. It is needed for aggregating training/inference images into one image for 
    demonstation needs.

    Args:
        x_target (np.ndarray): x_target input image.
        x_source (np.ndarray): x_source input image.
        y (np.ndarray): y input image (segmentation map).
        target_prediction (np.ndarray): prediction for x_target made by segmentation model.
        source_prediction (np.ndarray): prediction for x_source made by segmentation model.
        entropy_map (np.ndarray): entropy map made from self-information vector made by segmentation model.
    Returns:
        agg_img (np.ndarray): aggregated image with labels and legend.
    """
    fig, (ax_x_target, ax_x_source, ax_y, ax_target_pred, ax_source_pred, ax_ent_map) = plt.subplots(1, 6, figsize=(12, 2))
    fig.suptitle("")
    fig.tight_layout()

    ax_x_target.imshow(x_target.astype(float), cmap='binary')
    ax_x_target.axis("off")
    ax_x_target.set_title("x_target")

    ax_x_source.imshow(x_source.astype(float), cmap='binary')
    ax_x_source.axis("off")
    ax_x_source.set_title("x_source")

    ax_y.imshow(y.astype(float), cmap='jet')
    ax_y.axis("off")
    ax_y.set_title("y")

    ax_target_pred.imshow(target_prediction.astype(float), cmap='jet')
    ax_target_pred.axis("off")
    ax_target_pred.set_title("target_prediction")

    ax_source_pred.imshow(source_prediction.astype(float), cmap='jet')
    ax_source_pred.axis("off")
    ax_source_pred.set_title("source_prediction")

    ax_ent_map.imshow(entropy_map.astype(float), cmap='Blues')
    ax_ent_map.axis("off")
    ax_ent_map.set_title("entropy_map (target)")

    return fig
