import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def plot_mask_comparison(
    val_img: torch.Tensor,
    val_mask: torch.Tensor,
    val_pred: torch.Tensor,
    num_classes: int = 2,
) -> plt.Figure:
    """
    Plots an image, its ground truth mask, and predicted mask side by side.
    
    Args:
        val_img (torch.Tensor): Image tensor of shape (1, H, W) or (3, H, W).
        val_mask (torch.Tensor): Ground truth mask of shape (H, W).
        val_pred (torch.Tensor): Predicted mask of shape (H, W) or (C, H, W).
        num_classes (int): Number of classes in segmentation.
    """
    assert val_img.shape[0] in [1, 3] and val_img.ndim == 3
    assert val_mask.ndim == 2
    assert val_pred.ndim in [2, 3]

    img_np = val_img.permute(1, 2, 0).detach().cpu().numpy()
    mask_np = val_mask.detach().cpu().numpy()
    pred_np = val_pred.squeeze().detach().cpu().numpy()

    # If prediction is 3D, convert to class indices
    if val_pred.ndim == 3:
        pred_np = torch.argmax(val_pred, dim=0).detach().cpu().numpy()

    # Generate colormap for num_classes
    cmap = plt.get_cmap('tab10', num_classes)
    color_map = (cmap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)
    color_map[0] = [0, 0, 0] # set color of bg (class #0) as black [rgb 0,0,0]

    def colorize(mask):
        return color_map[mask.astype(np.uint8)]

    fig = plt.figure(figsize=(8, 3), dpi=80)

    # Image
    plt.subplot(1, 3, 1)
    if val_img.shape[0] == 1:
        plt.imshow(img_np.squeeze(), cmap='gray')
    else:
        plt.imshow(img_np)
    plt.title("SAR")
    plt.axis("off")

    # Ground Truth
    plt.subplot(1, 3, 2)
    plt.imshow(colorize(mask_np))
    plt.title("Ground Truth")
    plt.axis("off")

    # Prediction
    plt.subplot(1, 3, 3)
    plt.imshow(colorize(pred_np))
    plt.title("Predicted Mask")
    plt.axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=color_map[i] / 255.0, 
                    #    label=f"Class {i}", 
                       label='oil' if i==1 else 'background')
        for i in range(num_classes)
    ]
    plt.legend(
        handles=legend_patches,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    plt.tight_layout()
    return fig


def plot_to_tensorboard(
    writer: SummaryWriter, fig: plt.Figure, step: int, name: str
):
    """
    Converts a matplotlib figure to an image and writes it to tensorboard.
    """
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    img = img[:, :, :3]  # RGBA to RGB
    img = img / 255.0
    writer.add_image(name, img, step, dataformats="HWC")
    writer.flush()
    plt.close(fig)


def plot_tensorboard_batch_images(
    writer: SummaryWriter,
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    epoch: int = 0,
    name_tensorboard: str = "train",
    n_images: int = 5,
    num_classes: int = 2,
) -> None:
    """
    Plots batch of images, ground truth, and predictions to TensorBoard.
    
    Args:
        writer (SummaryWriter): TensorBoard writer.
        images (torch.Tensor): Shape (B, C, H, W).
        masks (torch.Tensor): Shape (B, H, W).
        predictions (torch.Tensor): Shape (B, H, W) or (B, C, H, W).
        epoch (int): Current epoch.
        name_tensorboard (str): Name for TensorBoard log.
        n_images (int): Number of images to plot.
        num_classes (int): Number of segmentation classes.
    """
    assert images.shape[0] == masks.shape[0] == predictions.shape[0]
    assert images.ndim == 4 and masks.ndim == 3 and predictions.ndim in [3, 4]
    assert images.shape[1] in [1, 3]

    n_images = min(images.shape[0], n_images)
    for idx_in_batch in range(n_images):
        fig = plot_mask_comparison(
            images[idx_in_batch],
            masks[idx_in_batch],
            predictions[idx_in_batch],
            num_classes=num_classes,
        )
        plot_to_tensorboard(
            writer,
            fig,
            idx_in_batch,
            f"{name_tensorboard}/Images_epoch_{epoch+1}"
        )
