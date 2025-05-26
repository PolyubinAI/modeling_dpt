import cv2
import numpy as np
from torchvision.transforms import Compose
from constants import DEPTH, MASK, IMAGE

# Literal constant
DISPARITY = "disparity"


class NormalizeImage(object):
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std

    def __call__(self, sample):
        sample[IMAGE] = (sample[IMAGE] - self._mean) / self._std
        return sample


class Resize(object):
    """Resize sample to given size (width, height)."""
    def __init__(  # noqa: WPS211
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):   # noqa: RST201, RST301
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                " lower_bound ": Output will be at least as large as the given size.
                " upper_bound ": Output will be at max as large as the given size.
                                (Output size might be smaller than given size.)
                " minimal ": Scale as least as possible.  (Output size might be smaller than given size.) # noqa: RST201, RST301
        """
        self._width = width
        self._height = height

        self._resize_target = resize_target
        self._keep_aspect_ratio = keep_aspect_ratio
        self._multiple_of = ensure_multiple_of
        self._resize_method = resize_method
        self._image_interpolation_method = image_interpolation_method

    def __call__(self, sample):
        width, height = self.get_size(  # noqa: WPS204
            sample[IMAGE].shape[1],
            sample[IMAGE].shape[0],
        )

        sample[IMAGE] = cv2.resize(  # noqa: WPS317
            sample[IMAGE],
            (width, height),
            interpolation=self._image_interpolation_method,
        )

        if self._resize_target:
            if DISPARITY in sample:
                sample[DISPARITY] = cv2.resize(  # noqa: WPS317
                    sample[DISPARITY],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

            if DEPTH in sample:
                sample[DEPTH] = cv2.resize(
                    sample[DEPTH], (width, height), interpolation=cv2.INTER_NEAREST,
                )

            sample[MASK] = cv2.resize(  # noqa: WPS317
                sample[MASK].astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            sample[MASK] = sample[MASK].astype(bool)

        return sample

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = np.round(x / self._multiple_of) * self._multiple_of
        y = y.astype(int)
        if max_val is not None and y > max_val:
            y = (np.floor(x / self._multiple_of) * self._multiple_of)
            y = y.astype(int)
        if y < min_val:
            y = (np.ceil(x / self._multiple_of) * self._multiple_of)
            y = y.astype(int)

        return y

    def get_size(self, width, height):  # noqa: WPS231
        # determine new height and width
        scale_height = self._height / height
        scale_width = self._width / width

        if self._keep_aspect_ratio:
            if self._resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self._resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self._resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self._resize_method} not implemented ",
                )

        if self._resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self._height,
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self._width,
            )
        elif self._resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self._height,
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self._width,
            )
        elif self._resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self._resize_method} not implemented")

        return (new_width, new_height)


class PrepareForNet(object):
    """Prepare sample for usage as network input."""

    def __call__(self, sample):        
        image = np.transpose(sample[IMAGE], (2, 0, 1))
        sample[IMAGE] = np.ascontiguousarray(image).astype(np.float32)

        if MASK in sample:
            sample[MASK] = sample[MASK].astype(np.float32)
            sample[MASK] = np.ascontiguousarray(sample[MASK])

        if DISPARITY in sample:
            disparity = sample[DISPARITY].astype(np.float32)
            sample[DISPARITY] = np.ascontiguousarray(disparity)

        if DEPTH in sample:
            depth = sample[DEPTH].astype(np.float32)
            sample[DEPTH] = np.ascontiguousarray(depth)

        return sample


def get_preprocess(dataset_name=None) -> Compose:

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
            NormalizeImage(mean=mean, std=std),
            PrepareForNet(),
        ],
    )
