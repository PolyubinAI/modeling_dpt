from pytorch_lightning.callbacks import Callback
import torch
from constants import LOW_CLIP_VALUE
import matplotlib.pyplot as plt  # noqa: WPS301
from typing import Union
import numpy as np


def min_max_scale(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Perform min-max scaling on a tensor to a range of 0 to 1.

    Parameters:
    x (torch.Tensor): The input tensor.

    Returns:
    torch.Tensor: The scaled tensor.
    """
    bias = x - torch.min(x)
    diff = torch.max(x) - torch.min(x) + LOW_CLIP_VALUE
    return bias / diff


def plt_figure(  # noqa: WPS213
    img: torch.Tensor,
    predictions: torch.Tensor,
    target: torch.Tensor,
    dataset_mode: str,
    idx: int = None,
) -> None:
    """
    Save predicted, target, and original images by concatenating them along dimension 1.

    Parameters:
    img (torch.Tensor): The original image tensor.
    predictions (torch.Tensor): The predicted image tensor.
    target (torch.Tensor): The target image tensor.
    save_path (str): The path where the output image will be saved.

    Returns:
    None
    """
    target = target.cpu().detach().numpy()
    predictions = predictions.cpu().detach().numpy()
    if isinstance(img, torch.Tensor):
        img = min_max_scale(img)
        img = img.cpu().detach().numpy()
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 3, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{dataset_mode}_vanilla image")
    fig.add_subplot(1, 3, 2)
    plt.imshow(predictions, cmap='terrain', vmin=0, vmax=None)
    plt.colorbar(aspect=5)
    plt.axis("off")
    plt.title(f"{dataset_mode}_predictions")
    fig.add_subplot(1, 3, 3)
    plt.imshow(target, cmap='terrain', vmin=0, vmax=None)
    plt.colorbar(aspect=5)
    plt.axis("off")
    plt.title(f"{dataset_mode}_target")
    
    if idx is None:
        idx = -1 
    
    plt.savefig(f'/home/apolyubin/private_data/logs_folder/diploma/custom_dpt/vaih/{dataset_mode}_{idx}.png', bbox_inches='tight', dpi=300)

    plt.show()


    plt.close()


class PredictAfterValidationCallback(Callback):
    def __init__(self, logger, config):
        super().__init__()
        self.logger = logger
        self.config = config

    def setup(self, trainer, pl_module, stage):
        if stage in ("fit", "validate"):  # noqa: WPS510
            # setup the predict data even for fit/validate, as we will call it during `on_validation_epoch_end`
            trainer.datamodule.setup("predict")

    def on_validation_epoch_end(self, trainer, pl_module):  # noqa: WPS210
        if trainer.sanity_checking:  # optional skip
            return
        
        val_dataloaders = trainer.datamodule.val_dataloader()

        # for idx, dataloader in enumerate(val_dataloader):
        batches = [pl_module.transfer_batch_to_device(
            next(iter(loader)),
            trainer.strategy.root_device,
            1,
        ) for loader in val_dataloaders]
        outputs = [pl_module.predict_step(batch) for batch in batches]
        for idx_outer, (output, batch) in enumerate(zip(outputs, batches)):
            inputs, mask = batch
            for idx in range(len(output)):
                plt_figure(
                    inputs[idx].permute(1, 2, 0),
                    output[idx],
                    mask[idx],
                    dataset_mode=f"val_{self.config.data_config.dataset_names[idx_outer]}",
                    idx=idx,
                )


class PredictAfterTrainingCallback(Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def setup(self, trainer, pl_module, stage):
        if stage in ("fit", "validate"):  # noqa: WPS510
            # setup the predict data even for fit/validate, as we will call it during `on_validation_epoch_end`
            trainer.datamodule.setup("predict")

    def on_train_epoch_end(self, trainer, pl_module):  # noqa: WPS210
        if trainer.sanity_checking:  # optional skip
            return

        train_dataloader = trainer.datamodule.train_dataloader()
        batch = pl_module.transfer_batch_to_device(
            next(iter(train_dataloader)),
            trainer.strategy.root_device,
            1,
        )
        outputs = pl_module.predict_step(batch)
        inputs, depth_masks = batch
        for i, out in enumerate(outputs):
            plt_figure(
                inputs[i].permute(1, 2, 0),
                out,
                depth_masks[i],
                dataset_mode="train",
                idx=i,
            )
