"""Augment each shape once, then expand its binary part targets."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch

from utils.aug import (
    CenterShift,
    ChromaticAutoContrast,
    ChromaticJitter,
    ChromaticTranslation,
    NormalizeColor,
    NormalizeMy,
    RandomRotate,
    RandomScale,
    ToTensor,
)


@contextmanager
def _random_seed(seed: int | None) -> Iterator[None]:
    if seed is None:
        yield
        return
    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _arrays(sample: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    missing = {"coords", "color", "normal", "gt_masks"} - set(sample)
    if missing:
        raise ValueError(f"sample is missing: {', '.join(sorted(missing))}")
    coords = np.asarray(sample["coords"], dtype=np.float32)
    color = np.asarray(sample["color"], dtype=np.float32)
    normal = np.asarray(sample["normal"], dtype=np.float32)
    masks = np.asarray(sample["gt_masks"], dtype=np.bool_)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        raise ValueError("coords must have shape [N, 3]")
    if color.shape != coords.shape or normal.shape != coords.shape:
        raise ValueError("color and normal must match coords")
    if masks.ndim != 2 or masks.shape[1] != len(coords) or len(masks) == 0:
        raise ValueError("gt_masks must have shape [M, N]")
    return coords.copy(), color.copy(), normal.copy(), masks.copy()


def prepare_shape(
    sample: Mapping[str, Any], *, training: bool, seed: int | None = None
) -> dict[str, torch.Tensor]:
    """Apply one shared transform to all targets from a shape."""

    coords, color, normal, _ = _arrays(sample)
    data: dict[str, Any] = {
        "coord": coords,
        "color": color,
        "normal": normal,
        # Existing PartSAM transforms update this key together with coordinates.
        "vertices": coords.copy(),
    }
    with _random_seed(seed):
        data = CenterShift(apply_z=True)(data)
        if training:
            for axis in ("z", "x", "y"):
                data = RandomRotate(angle=[-0.5, 0.5], axis=axis, center=[0, 0, 0], p=0.7)(data)
            for axis in (0, 1):
                if np.random.rand() < 0.5:
                    data["coord"][:, axis] *= -1
                    data["normal"][:, axis] *= -1
                    data["vertices"][:, axis] *= -1
            data = ChromaticAutoContrast(p=0.2, blend_factor=None)(data)
            data = ChromaticTranslation(p=0.6, ratio=0.05)(data)
            data = ChromaticJitter(p=0.6, std=0.05)(data)
            data = CenterShift(apply_z=True)(data)
        data = NormalizeMy()(data)
        if training:
            data = RandomScale(scale=[0.95, 1.05], anisotropic=False)(data)
        data = NormalizeColor()(data)
        data.pop("vertices")
        data = ToTensor()(data)
    return {
        "coords": data["coord"].float(),
        "color": data["color"].float(),
        "normal": data["normal"].float(),
    }


def collate_part_masks(
    batch: Sequence[Mapping[str, Any]],
    *,
    training: bool,
    max_targets: int | None = 4,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Expand shape batches to one model item per binary part mask."""

    if not batch:
        raise ValueError("batch must not be empty")
    records: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = []
    point_count: int | None = None
    for shape_index, sample in enumerate(batch):
        coords, _, _, masks = _arrays(sample)
        if point_count is None:
            point_count = len(coords)
        elif len(coords) != point_count:
            raise ValueError("all shapes in a batch must have the same point count")
        shape_seed = None if seed is None else seed + shape_index
        prepared = prepare_shape(sample, training=training, seed=shape_seed)
        for mask in torch.from_numpy(masks):
            records.append((prepared, mask.unsqueeze(0)))

    if max_targets is not None:
        if max_targets <= 0:
            raise ValueError("max_targets must be positive or None")
        if len(records) > max_targets:
            generator = None
            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(len(records), generator=generator)[:max_targets]
            records = [records[int(index)] for index in order]

    return {
        "coords": torch.stack([record[0]["coords"] for record in records]),
        "color": torch.stack([record[0]["color"] for record in records]),
        "normal": torch.stack([record[0]["normal"] for record in records]),
        "gt_masks": torch.stack([record[1] for record in records]).bool(),
    }


class PartMaskCollator:
    def __init__(self, training: bool, max_targets: int | None = 4) -> None:
        self.training = bool(training)
        self.max_targets = max_targets

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        return collate_part_masks(
            batch, training=self.training, max_targets=self.max_targets
        )


__all__ = ["PartMaskCollator", "collate_part_masks", "prepare_shape"]
