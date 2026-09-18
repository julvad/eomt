import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def plot_binary_mask_comparison(
    val_img: torch.Tensor,
    val_mask: torch.Tensor,
    val_pred: torch.Tensor,
) -> plt.Figure:
    """
    Plots an image, its binary ground truth mask, and predicted mask side by side.

    Args:
        val_img (torch.Tensor): Image tensor of shape (1, H, W) or (3, H, W).
        val_mask (torch.Tensor): Ground truth binary mask of shape (H, W).
        val_pred (torch.Tensor): Predicted binary mask of shape (1, H, W).
    """
    assert val_img.shape[0] in [1, 3] and val_img.ndim == 3, 'Error with val_img.shape[0] in [1, 3] and val_img.ndim == 3'
    assert val_mask.ndim == 2, 'error with val_mask.ndim == 2'
    assert val_pred.shape[0] == 1 and val_pred.ndim == 3, 'Error with val_pred.shape[0] == 1 and val_pred.ndim == 3'

    img_np = val_img.permute(1, 2, 0).detach().cpu().numpy()
    mask_np = val_mask.squeeze().detach().cpu().numpy()
    pred_np = val_pred.squeeze().detach().cpu().numpy()

    # Define colors: 0=black, 1=red
    color_map = np.array([[0, 0, 0], [255, 0, 0]], dtype=np.uint8)

    def colorize(mask):
        return color_map[mask.astype(np.uint8)]

    # fig = plt.figure(figsize=(15, 5))
    fig = plt.figure(figsize=(8, 3), dpi=80) # reduce resolution to try to avoid memory errors (i.e. Failed to allocate bitmap)

    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("SAR")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(colorize(mask_np))
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(colorize(pred_np))
    plt.title("Predicted Mask")
    plt.axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=np.array([0, 0, 0]) / 255, label="Background"),
        mpatches.Patch(color=np.array([255, 0, 0]) / 255, label="Oil slick"),
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
    writer: SummaryWriter, fig: matplotlib.figure.Figure, step: int, name: str
):
    """
    Converts a matplotlib figure to an image and writes it to tensorboard.

    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter instance.
        fig (matplotlib.figure.Figure): Matplotlib figure to save.
        step (int): Global step value to record.
        name (str): Name for the image in tensorboard.
    """
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    # Convert RGBA to RGB
    img = img[:, :, :3] # jv: not needed for SAR? Images already converted to rgb?
    img = img / 255.0
    # print("Image shape:", img.shape, "dtype:", img.dtype, "min/max:", img.min(), img.max()) ## troubleshoot
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
    n_images:int=5
) -> None:
    """

    args:
        images: torch.Tensor of shape (batch_size, (1 | 3), height, width)
        masks: torch.Tensor of shape (batch_size, height, width)
        predictions: torch.Tensor of shape (batch_size, height, width)
        epoch: int
    """
    assert images.shape[0] == masks.shape[0] == predictions.shape[0], f'images.shape[0] != masks.shape[0] != predictions.shape[0]'
    assert images.ndim == 4 and masks.ndim == 3 and predictions.ndim == 3, f'images.ndim !=4 or masks.ndim !=3 or predictions.ndim!=3'
    assert images.shape[1] in [1, 3], f'images.shape[1] not in [1,3]: {images.shape[1]}'
    h, w = images.shape[2], images.shape[3]
    # Sometimes masks and predictions do not exactly match the image size
    assert masks.shape[1] == h and masks.shape[2] == w, 'masks shape not match img shape'
    assert predictions.shape[1] == h and predictions.shape[2] == w, 'masks shape not match img shape'


    # for idx_in_batch in range(images.shape[0]): #original oceansar
    n_images = min(images.shape[0], n_images) # if batch is lower than n_images
    for idx_in_batch in range(n_images):
        # print("Image:", images[idx_in_batch].shape)
        # print("Mask:", masks[idx_in_batch].shape)
        # print("Prediction:", predictions[idx_in_batch].shape) # troubleshoot
            # fig = plot_binary_mask_comparison(
            #     images[idx_in_batch, :],
            #     masks[idx_in_batch, :],
            #     predictions[idx_in_batch, :].unsqueeze(0), # original oceansar
            # )
        fig = plot_binary_mask_comparison(
            images[idx_in_batch],                      # [3, 512, 512]
            masks[idx_in_batch],                      # [512, 512]
            predictions[idx_in_batch].unsqueeze(0),   # [1, 512, 512]
        )
        # Log validation images
        plot_to_tensorboard(
            writer,
            fig,
            idx_in_batch,
            f"{name_tensorboard}/Images_epoch_{epoch}"
        )
