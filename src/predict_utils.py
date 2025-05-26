import torch
from constants import LOW_CLIP_VALUE, HIGH_CLIP_VALUE


def make_prediction_dpt(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
    """
    Make depth predictions using the Model.

    Args:
        model (torch.nn.Module): The Model.
        img (torch.Tensor): The input image tensor.

    Returns:
        torch.Tensor: The predicted depth tensor.
    """
    

    disparity_image = model(img).predicted_depth
    disparity_image = disparity_image.reshape(
        disparity_image.shape[0],
        1,
        disparity_image.shape[1],
        disparity_image.shape[2],
    )

    ans = torch.nn.functional.interpolate(
        disparity_image,
        size=tuple(img.shape[2:]),
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)
    
    return torch.clip(ans, LOW_CLIP_VALUE, HIGH_CLIP_VALUE)
