from configs.config import Config
config = Config.from_yaml('./configs/config.yaml')

import os
os.environ["CUDA_VISIBLE_DEVICES"]=config.device

import numpy as np
import torch
from empatches import EMPatches
from PIL import Image
import albumentations as albu
from torchvision.transforms import Compose, Normalize
from transformers import DPTForDepthEstimation
Image.MAX_IMAGE_PIXELS = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOW_CLIP_VALUE = 1e-8
HIGH_CLIP_VALUE = 1000
DEPTH = "depth"
MASK = "mask"
IMAGE = "image"


def make_prediction_dpt(model: torch.nn.Module, img: torch.Tensor) -> torch.Tensor:
    """
    Make depth predictions using the Model.

    Args:
        model (torch.nn.Module): The Model.
        img (torch.Tensor): The input image tensor.

    Returns:
        torch.Tensor: The predicted depth tensor.
    """
    if img.ndim != 4:
        img = img.unsqueeze(0)
    if img.shape != (384, 384):
        img = torch.nn.functional.interpolate(
        img,
        size=(384, 384),
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)
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


class PrepareForNet(object):
    """Prepare sample for usage as network input."""

    def __call__(self, sample):        
        image = np.transpose(sample[IMAGE], (2, 0, 1))
        sample[IMAGE] = np.ascontiguousarray(image).astype(np.float32)

        if MASK in sample:
            sample[MASK] = sample[MASK].astype(np.float32)
            sample[MASK] = np.ascontiguousarray(sample[MASK])

        if DEPTH in sample:
            depth = sample[DEPTH].astype(np.float32)
            sample[DEPTH] = np.ascontiguousarray(depth)

        return sample


def get_preprocess(dataset_name=None) -> albu.Compose:
    """
    Returns:
        Compose: The composed transformation pipeline.
    """

    if dataset_name == 'DFC2018':
        mean = (0.47201676, 0.32118613, 0.3189363)
        std = (0.21518334, 0.15455843, 0.14943491)
    elif dataset_name == 'Vaihingen':
        mean = (0.4643, 0.3185, 0.3141)
        std = (0.2171, 0.1561, 0.1496)
    elif dataset_name == 'Potsdam':
        mean = (0.51882998, 0.51636622, 0.40003074)
        std = (0.25592191, 0.22129431, 0.1813154)
    if dataset_name is None:
        mean = (0.1559, 0.1109, 0.1098)
        std = (0.2536, 0.1841, 0.1813)

    return Compose(
        [
            Normalize(mean=mean, std=std),
        ],
    )


def predict_large_depth_map(img, model, patch_size=384, overlap_size=0.5):
    """
    Make depth predictions of large Satelite by the Model.

    Args:
        img (torch.Tensor) : Normalized Satelite Image.
        model (torch.nn.Module): The Model.
        patch_size (int): size of crop patch.
        overlap_size (float): crop overlapping cofficient

    Returns:
        torch.Tensor: The predicted depth tensor.
    """
    emp = EMPatches()
    if img.shape[0]:
        img = img.permute(1, 2, 0)
    img_patches, indices = emp.extract_patches(img, patchsize=patch_size, overlap=overlap_size)
    preds = []
    for img in img_patches:
        img = img.permute(2, 0, 1).unsqueeze(0)
        pred = make_prediction_dpt(model, img)
        preds.append((pred.permute(1, 2, 0).squeeze()).detach().cpu())
    merged_img = emp.merge_patches(preds, indices, mode='avg')
    return merged_img

def load_and_preprocces_dfc_image(image_path):
    """
    load and preprocces dfc image for prediction by DPT

    Args:
        img_path (str): Path to load the image.

    Returns:
        torch.Tensor: The image tensor.
    """
    img = Image.open(image_path).convert("RGB")
    img = torch.tensor(np.array(img, dtype=np.float32))
    img = img.permute(2, 0, 1)
    a = get_preprocess('DFC2018')
    img = a(img) / 255.
    img = torch.Tensor(img)

    return img

if __name__ == '__main__':
    img= load_and_preprocces_dfc_image("../data/DFC2018/og_data/RGB/UH_NAD83_272056_3290290.tif")

    model = DPTForDepthEstimation.from_pretrained('<Path_to_your_model>')

    merged_img = predict_large_depth_map(img, model, patch_size=384, overlap_size=0.5)

    im = Image.fromarray(
    merged_img,
    ).convert('RGB')

    im.save('../logs_folder/inference_logs/preds.jpg')

