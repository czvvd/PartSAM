"""Utilities for initializing PartSAM's two PartField encoder branches."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Mapping, Union

import torch
import torch.nn as nn


_STATE_DICT_WRAPPERS = ("state_dict", "model_state_dict", "model", "weights")


@dataclass(frozen=True)
class PartFieldLoadReport:
    """A deterministic summary of a PartField checkpoint load."""

    matched: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatched: tuple[str, ...]


class PartFieldCheckpointError(RuntimeError):
    """Raised when a checkpoint cannot initialize both required branches."""

    def __init__(self, message: str, report: PartFieldLoadReport | None = None):
        super().__init__(message)
        self.report = report or PartFieldLoadReport((), (), (), ())


Checkpoint = Union[str, PathLike[str], Mapping[str, object]]


def _load_checkpoint(checkpoint: Checkpoint) -> Mapping[str, object]:
    if isinstance(checkpoint, (str, PathLike)):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"PartField checkpoint does not exist: {path}")
        if path.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file

            checkpoint = load_file(str(path), device="cpu")
        else:
            # The published PartField Lightning checkpoint contains its training
            # configuration as well as tensors, so it is not a weights-only file.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise PartFieldCheckpointError(
            "PartField checkpoint must be a path or a mapping containing a tensor state dict"
        )
    return checkpoint


def _extract_state_dict(checkpoint: Mapping[str, object]) -> Mapping[str, torch.Tensor]:
    current: Mapping[str, object] = checkpoint
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        nested = next(
            (
                current[key]
                for key in _STATE_DICT_WRAPPERS
                if key in current and isinstance(current[key], Mapping)
            ),
            None,
        )
        if nested is None:
            break
        current = nested

    state_dict = {
        str(key): value for key, value in current.items() if torch.is_tensor(value)
    }
    if not state_dict:
        raise PartFieldCheckpointError(
            "PartField checkpoint does not contain a tensor state dict"
        )
    return state_dict


def _normalise_state_dict(
    source: Mapping[str, torch.Tensor], target_keys: set[str]
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    normalised: dict[str, torch.Tensor] = {}
    original_keys: dict[str, str] = {}
    unexpected: list[str] = []

    for source_key, value in source.items():
        components = source_key.split(".")
        candidates = [".".join(components[offset:]) for offset in range(len(components))]
        target_key = next((key for key in candidates if key in target_keys), None)
        if target_key is None:
            unexpected.append(source_key)
            continue
        if target_key in normalised:
            first = original_keys[target_key]
            raise PartFieldCheckpointError(
                "PartField checkpoint has multiple keys that resolve to "
                f"{target_key!r}: {first!r} and {source_key!r}"
            )
        normalised[target_key] = value
        original_keys[target_key] = source_key

    return normalised, tuple(sorted(unexpected))


def _looks_like_partsam_state(source: Mapping[str, torch.Tensor]) -> bool:
    partsam_components = {
        "partfieldMy",
        "mask_decoder",
        "mask_encoder",
        "prompt_encoder",
        "prompt_point_mapper",
    }
    return any(
        partsam_components.intersection(source_key.split("."))
        for source_key in source
    )


def load_partfield_branches(
    reference: nn.Module,
    trainable: nn.Module,
    checkpoint: Checkpoint,
) -> PartFieldLoadReport:
    """Load the published PartField weights into both encoder branches.

    The reference branch must be an exact architectural match. The trainable
    branch may have additional adaptation layers; those remain freshly
    initialized and are listed in ``report.missing``.
    """

    checkpoint_mapping = _load_checkpoint(checkpoint)
    source = _extract_state_dict(checkpoint_mapping)
    if _looks_like_partsam_state(source):
        raise PartFieldCheckpointError(
            "PartSAM checkpoint cannot initialize a new run. Configure the public "
            "PartField checkpoint for initialization, or use train.resume_from to "
            "resume a checkpoint produced by the same run."
        )
    reference_state = reference.state_dict()
    trainable_state = trainable.state_dict()
    target_keys = set(reference_state) | set(trainable_state)
    normalised, unexpected = _normalise_state_dict(source, target_keys)

    selected: dict[str, dict[str, torch.Tensor]] = {
        "partfield": {},
        "partfieldMy": {},
    }
    matched: list[str] = []
    missing: list[str] = []
    shape_mismatched: list[str] = []

    for branch_name, branch_state in (
        ("partfield", reference_state),
        ("partfieldMy", trainable_state),
    ):
        for key, target in branch_state.items():
            qualified_key = f"{branch_name}.{key}"
            value = normalised.get(key)
            if value is None:
                missing.append(qualified_key)
                continue
            if value.shape != target.shape:
                shape_mismatched.append(
                    f"{qualified_key}: checkpoint {tuple(value.shape)} != model {tuple(target.shape)}"
                )
                continue
            selected[branch_name][key] = value
            matched.append(qualified_key)

    report = PartFieldLoadReport(
        matched=tuple(sorted(matched)),
        missing=tuple(sorted(missing)),
        unexpected=unexpected,
        shape_mismatched=tuple(sorted(shape_mismatched)),
    )

    required_missing = [
        key
        for key in report.missing
        if key.startswith("partfield.")
        or key.removeprefix("partfieldMy.") in reference_state
    ]
    if report.shape_mismatched:
        raise PartFieldCheckpointError(
            "PartField checkpoint shape mismatch: "
            + "; ".join(report.shape_mismatched),
            report,
        )
    if required_missing:
        raise PartFieldCheckpointError(
            "PartField checkpoint is missing required weights: "
            + ", ".join(required_missing),
            report,
        )

    # Validation above is deliberately transactional: neither branch is
    # modified until all required tensors have been checked.
    reference.load_state_dict(selected["partfield"], strict=True)
    trainable.load_state_dict(selected["partfieldMy"], strict=False)
    return report


class PartFieldDualInitializationMixin:
    """Checkpoint initialization and train-mode policy for a dual encoder."""

    partfield: nn.Module
    partfieldMy: nn.Module

    def load_partfield_checkpoint(self, checkpoint: Checkpoint) -> PartFieldLoadReport:
        report = load_partfield_branches(self.partfield, self.partfieldMy, checkpoint)
        self.partfield.requires_grad_(False)
        self.partfield.eval()
        self.partfieldMy.requires_grad_(True)
        return report

    def train(self, mode: bool = True):
        module = super().train(mode)
        # ``nn.Module.train`` is recursive, so restore the frozen reference
        # branch to eval mode every time the parent mode changes.
        self.partfield.eval()
        return module
