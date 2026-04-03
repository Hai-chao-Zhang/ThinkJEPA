# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
# Copyright (c) 2024-2026 Northeastern University.
# Developed in NEU SMILE LAB by Haichao Zhang (https://zhanghaichao.xyz)
# and Yun Raymond Fu (https://www1.ece.neu.edu/~yunfu/).
# SPDX-style identifier: LicenseRef-ThinkJEPA-Attribution
# Original source: https://github.com/Hai-chao-Zhang/ThinkJEPA
# See the root LICENSE, NOTICE, CITATION.cff, and CITATION.bib for attribution and citation requirements.

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_RGB_MEAN = (0.485, 0.456, 0.406)
DEFAULT_RGB_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SpatialAdaptationRecipe:
    short_side: int
    crop_height: int
    crop_width: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def _coerce_crop_size(size_attr):
    if isinstance(size_attr, int):
        return int(size_attr), int(size_attr)
    if isinstance(size_attr, (tuple, list)) and len(size_attr) == 2:
        return int(size_attr[0]), int(size_attr[1])
    raise ValueError(f"Unsupported crop size specification: {size_attr}")


@torch.no_grad()
def _read_spatial_recipe(eval_transform) -> SpatialAdaptationRecipe:
    detected_short_side = None
    detected_crop = None
    detected_mean = None
    detected_std = None

    for transform_op in getattr(eval_transform, "transforms", []):
        op_name = transform_op.__class__.__name__.lower()
        if "resize" in op_name and detected_short_side is None:
            resize_size = getattr(transform_op, "size", None)
            if isinstance(resize_size, int):
                detected_short_side = int(resize_size)
            elif isinstance(resize_size, (tuple, list)) and len(resize_size) == 2:
                detected_short_side = int(min(resize_size))
            else:
                raise ValueError(f"Unsupported resize size specification: {resize_size}")
        elif "centercrop" in op_name and detected_crop is None:
            detected_crop = _coerce_crop_size(getattr(transform_op, "size", None))
        elif "normalize" in op_name and detected_mean is None and detected_std is None:
            detected_mean = tuple(float(value) for value in getattr(transform_op, "mean"))
            detected_std = tuple(float(value) for value in getattr(transform_op, "std"))

    if detected_short_side is None or detected_crop is None:
        raise RuntimeError("Could not infer resize and center-crop settings from eval_transform.")

    if detected_mean is None or detected_std is None:
        detected_mean = DEFAULT_RGB_MEAN
        detected_std = DEFAULT_RGB_STD

    crop_height, crop_width = detected_crop
    return SpatialAdaptationRecipe(
        short_side=detected_short_side,
        crop_height=crop_height,
        crop_width=crop_width,
        mean=detected_mean,
        std=detected_std,
    )


class VideoObservationAdapter(nn.Module):
    """Convert raw video tensors into the normalized layout expected by the JEPA encoder."""

    def __init__(self, eval_transform, antialias: bool = False, device: str | None = None):
        super().__init__()
        del device
        self.antialias = antialias
        recipe = _read_spatial_recipe(eval_transform)
        self.target_short_side = recipe.short_side
        self.crop_height = recipe.crop_height
        self.crop_width = recipe.crop_width
        self.register_buffer(
            "channel_mean",
            torch.tensor(recipe.mean, dtype=torch.float32)[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "channel_std",
            torch.tensor(recipe.std, dtype=torch.float32)[None, :, None, None],
            persistent=False,
        )

    @staticmethod
    def _add_batch_axis(video_tensor):
        if video_tensor.ndim == 4:
            return video_tensor.unsqueeze(0), True
        if video_tensor.ndim == 5:
            return video_tensor, False
        raise ValueError(f"Expected a 4D or 5D video tensor, got shape {tuple(video_tensor.shape)}")

    @staticmethod
    def _trim_to_rgb(video_tensor):
        if video_tensor.shape[-1] not in (3, 4):
            raise ValueError(f"Expected RGB/RGBA channels in the last axis, got shape {tuple(video_tensor.shape)}")
        if video_tensor.shape[-1] == 4:
            return video_tensor[..., :3]
        return video_tensor

    @staticmethod
    def _to_unit_float(video_tensor):
        if not torch.is_floating_point(video_tensor):
            video_tensor = video_tensor.float()
        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0
        return video_tensor

    def _resize_short_side(self, frames):
        _, _, height, width = frames.shape
        if height <= width:
            resized_height = self.target_short_side
            resized_width = int(round(width * (self.target_short_side / height)))
        else:
            resized_height = int(round(height * (self.target_short_side / width)))
            resized_width = self.target_short_side
        resized_frames = F.interpolate(
            frames,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=self.antialias,
        )
        return resized_frames, resized_height, resized_width

    def _center_crop(self, frames, resized_height, resized_width):
        crop_top = int(round((resized_height - self.crop_height) / 2.0))
        crop_left = int(round((resized_width - self.crop_width) / 2.0))
        if crop_top < 0 or crop_left < 0:
            raise ValueError(
                f"Crop size {(self.crop_height, self.crop_width)} exceeds resized resolution {(resized_height, resized_width)}"
            )
        return frames[
            :,
            :,
            crop_top : crop_top + self.crop_height,
            crop_left : crop_left + self.crop_width,
        ]

    def forward(self, video_tensor):
        video_tensor, squeezed_batch = self._add_batch_axis(video_tensor)
        video_tensor = self._trim_to_rgb(video_tensor)
        video_tensor = self._to_unit_float(video_tensor)

        batch_size, num_frames, frame_height, frame_width, num_channels = video_tensor.shape
        channels_first = video_tensor.permute(0, 1, 4, 2, 3).contiguous()
        flattened_frames = channels_first.view(
            batch_size * num_frames, num_channels, frame_height, frame_width
        )

        resized_frames, resized_height, resized_width = self._resize_short_side(flattened_frames)
        cropped_frames = self._center_crop(
            resized_frames, resized_height=resized_height, resized_width=resized_width
        )

        mean = self.channel_mean.to(cropped_frames.device, non_blocking=True)
        std = self.channel_std.to(cropped_frames.device, non_blocking=True)
        normalized_frames = (cropped_frames - mean) / std
        normalized_video = normalized_frames.view(
            batch_size, num_frames, num_channels, self.crop_height, self.crop_width
        )
        if squeezed_batch:
            return normalized_video.squeeze(0)
        return normalized_video
