import math
import numbers
import warnings
from typing import Any, Callable, Dict, List, Tuple

import PIL.Image
import torch
from torch.nn.functional import one_hot
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import datapoints, transforms as _transforms
from torchvision.transforms import functional as F

from torchvision.transforms import _RandomApplyTransform, Transform
from ._utils import _parse_labels_getter
from ._utils import has_any, is_simple_tensor, query_chw, query_size


class _BaseMixUpCutMix(Transform):
    def __init__(self, *, alpha: float = 1.0, num_classes: int, labels_getter="default") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self._dist = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([alpha]))

        self.num_classes = num_classes

        self._labels_getter = _parse_labels_getter(labels_getter)

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, spec = tree_flatten(inputs)
        needs_transform_list = self._needs_transform_list(flat_inputs)

        if has_any(flat_inputs, PIL.Image.Image, datapoints.BoundingBoxes, datapoints.Mask):
            raise ValueError(f"{type(self).__name__}() does not support PIL images, bounding boxes and masks.")

        labels = self._labels_getter(inputs)
        if not isinstance(labels, torch.Tensor):
            raise ValueError(f"The labels must be a tensor, but got {type(labels)} instead.")
        elif labels.ndim != 1:
            raise ValueError(
                f"labels tensor should be of shape (batch_size,) " f"but got shape {labels.shape} instead."
            )

        params = {
            "labels": labels,
            "batch_size": labels.shape[0],
            **self._get_params(
                [inpt for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list) if needs_transform]
            ),
        }

        # By default, the labels will be False inside needs_transform_list, since they are a torch.Tensor coming
        # after an image or video. However, we need to handle them in _transform, so we make sure to set them to True
        needs_transform_list[next(idx for idx, inpt in enumerate(flat_inputs) if inpt is labels)] = True
        flat_outputs = [
            self._transform(inpt, params) if needs_transform else inpt
            for (inpt, needs_transform) in zip(flat_inputs, needs_transform_list)
        ]

        return tree_unflatten(flat_outputs, spec)

    def _check_image_or_video(self, inpt: torch.Tensor, *, batch_size: int):
        expected_num_dims = 5 if isinstance(inpt, datapoints.Video) else 4
        if inpt.ndim != expected_num_dims:
            raise ValueError(
                f"Expected a batched input with {expected_num_dims} dims, but got {inpt.ndim} dimensions instead."
            )
        if inpt.shape[0] != batch_size:
            raise ValueError(
                f"The batch size of the image or video does not match the batch size of the labels: "
                f"{inpt.shape[0]} != {batch_size}."
            )

    def _mixup_label(self, label: torch.Tensor, *, lam: float) -> torch.Tensor:
        label = one_hot(label, num_classes=self.num_classes)
        if not label.dtype.is_floating_point:
            label = label.float()
        return label.roll(1, 0).mul_(1.0 - lam).add_(label.mul(lam))


class MixUp(_BaseMixUpCutMix):
    """[BETA] Apply MixUp to the provided batch of images and labels.

    .. v2betastatus:: MixUp transform

    Paper: `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`_.

    .. note::
        This transform is meant to be used on **batches** of samples, not
        individual images. See
        :ref:`sphx_glr_auto_examples_plot_cutmix_mixup.py` for detailed usage
        examples.
        The sample pairing is deterministic and done by matching consecutive
        samples in the batch, so the batch needs to be shuffled (this is an
        implementation detail, not a guaranteed convention.)

    In the input, the labels are expected to be a tensor of shape ``(batch_size,)``. They will be transformed
    into a tensor of shape ``(batch_size, num_classes)``.

    Args:
        alpha (float, optional): hyperparameter of the Beta distribution used for mixup. Default is 1.
        num_classes (int): number of classes in the batch. Used for one-hot-encoding.
        labels_getter (callable or "default", optional): indicates how to identify the labels in the input.
            By default, this will pick the second parameter a the labels if it's a tensor. This covers the most
            common scenario where this transform is called as ``MixUp()(imgs_batch, labels_batch)``.
            It can also be a callable that takes the same input as the transform, and returns the labels.
    """

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict(lam=float(self._dist.sample(())))  # type: ignore[arg-type]

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        lam = params["lam"]

        if inpt is params["labels"]:
            return self._mixup_label(inpt, lam=lam)
        elif isinstance(inpt, (datapoints.Image, datapoints.Video)) or is_simple_tensor(inpt):
            self._check_image_or_video(inpt, batch_size=params["batch_size"])

            output = inpt.roll(1, 0).mul_(1.0 - lam).add_(inpt.mul(lam))

            if isinstance(inpt, (datapoints.Image, datapoints.Video)):
                output = type(inpt).wrap_like(inpt, output)  # type: ignore[arg-type]

            return output
        else:
            return inpt

class CutMix(_BaseMixUpCutMix):
    """[BETA] Apply CutMix to the provided batch of images and labels.

    .. v2betastatus:: CutMix transform

    Paper: `CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features
    <https://arxiv.org/abs/1905.04899>`_.

    .. note::
        This transform is meant to be used on **batches** of samples, not
        individual images. See
        :ref:`sphx_glr_auto_examples_plot_cutmix_mixup.py` for detailed usage
        examples.
        The sample pairing is deterministic and done by matching consecutive
        samples in the batch, so the batch needs to be shuffled (this is an
        implementation detail, not a guaranteed convention.)

    In the input, the labels are expected to be a tensor of shape ``(batch_size,)``. They will be transformed
    into a tensor of shape ``(batch_size, num_classes)``.

    Args:
        alpha (float, optional): hyperparameter of the Beta distribution used for mixup. Default is 1.
        num_classes (int): number of classes in the batch. Used for one-hot-encoding.
        labels_getter (callable or "default", optional): indicates how to identify the labels in the input.
            By default, this will pick the second parameter a the labels if it's a tensor. This covers the most
            common scenario where this transform is called as ``CutMix()(imgs_batch, labels_batch)``.
            It can also be a callable that takes the same input as the transform, and returns the labels.
    """

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        lam = float(self._dist.sample(()))  # type: ignore[arg-type]

        H, W = query_size(flat_inputs)

        r_x = torch.randint(W, size=(1,))
        r_y = torch.randint(H, size=(1,))

        r = 0.5 * math.sqrt(1.0 - lam)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))
        box = (x1, y1, x2, y2)

        lam_adjusted = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        return dict(box=box, lam_adjusted=lam_adjusted)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if inpt is params["labels"]:
            return self._mixup_label(inpt, lam=params["lam_adjusted"])
        elif isinstance(inpt, (datapoints.Image, datapoints.Video)) or is_simple_tensor(inpt):
            self._check_image_or_video(inpt, batch_size=params["batch_size"])

            x1, y1, x2, y2 = params["box"]
            rolled = inpt.roll(1, 0)
            output = inpt.clone()
            output[..., y1:y2, x1:x2] = rolled[..., y1:y2, x1:x2]

            if isinstance(inpt, (datapoints.Image, datapoints.Video)):
                output = inpt.wrap_like(inpt, output)  # type: ignore[arg-type]

            return output
        else:
            return inpt
