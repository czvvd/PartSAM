"""Minimal Accelerate training loop used by PartSAM."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR


@dataclass(frozen=True)
class TrainingResult:
    global_step: int
    checkpoints: tuple[Path, ...]


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    encoder_lr_scale: float = 0.2,
) -> AdamW:
    """Use the research learning-rate scale for the point encoder."""

    lr = float(lr)
    weight_decay = float(weight_decay)
    encoder_lr_scale = float(encoder_lr_scale)
    if lr <= 0 or encoder_lr_scale <= 0 or weight_decay < 0:
        raise ValueError("invalid optimizer settings")

    encoder = getattr(model, "pc_encoder", None)
    encoder_parameters = (
        [parameter for parameter in encoder.parameters() if parameter.requires_grad]
        if isinstance(encoder, nn.Module)
        else []
    )
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_ids
    ]

    groups: list[dict[str, Any]] = []
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": lr * encoder_lr_scale})
    if other_parameters:
        groups.append({"params": other_parameters, "lr": lr})
    if not groups:
        raise ValueError("model has no trainable parameters")
    return AdamW(groups, lr=lr, weight_decay=weight_decay)


def _model_inputs(batch: Mapping[str, Any]) -> tuple[torch.Tensor, ...]:
    names = ("coords", "color", "normal", "gt_masks")
    if any(name not in batch for name in names):
        raise ValueError("training batch must contain coords, color, normal and gt_masks")
    values = tuple(batch[name] for name in names)
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("all training inputs must be tensors")
    return values


def _loss_target(gt_masks: torch.Tensor) -> torch.Tensor:
    if gt_masks.ndim == 3 and gt_masks.shape[1] == 1:
        return gt_masks[:, 0]
    if gt_masks.ndim == 2:
        return gt_masks
    raise ValueError("gt_masks must have shape [B, 1, N] or [B, N]")


def _log_statistics(
    loss: torch.Tensor,
    auxiliary: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    """Return sums and counts for loss, first-round IoU and final-round IoU."""

    stats = loss.detach().new_zeros(6, dtype=torch.float32)
    stats[0] = loss.detach().float()
    stats[1] = 1
    interactions = [item for item in auxiliary if isinstance(item.get("iou"), torch.Tensor)]
    if interactions:
        first = interactions[0]["iou"].detach().float()
        final = interactions[-1]["iou"].detach().float()
        stats[2], stats[3] = first.sum(), first.numel()
        stats[4], stats[5] = final.sum(), final.numel()
    return stats


_CHECKPOINT_PATTERN = re.compile(r"^step-(\d{8})$")


def _checkpoints(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    return sorted(
        path
        for path in output_dir.iterdir()
        if path.is_dir() and _CHECKPOINT_PATTERN.fullmatch(path.name)
    )


def _write_state(checkpoint_dir: Path, global_step: int) -> None:
    temporary = checkpoint_dir / ".trainer_state.json.tmp"
    temporary.write_text(
        json.dumps({"global_step": global_step}) + "\n",
        encoding="utf-8",
    )
    temporary.replace(checkpoint_dir / "trainer_state.json")


def _read_state(checkpoint_dir: Path) -> int:
    state_path = checkpoint_dir / "trainer_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        global_step = int(state["global_step"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid trainer state: {state_path}") from error
    if global_step < 0:
        raise ValueError(f"invalid global_step in {state_path}")
    return global_step


def _resume_checkpoint(output_dir: Path, value: Any) -> tuple[Path, int]:
    """Accept only a complete step directory belonging to this output run."""

    checkpoint = Path(str(value)).expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    match = _CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    if checkpoint.parent != output_dir or match is None or not checkpoint.is_dir():
        raise ValueError(
            "train.resume_from must be a step checkpoint directly inside "
            "checkpoint.output_dir"
        )
    global_step = _read_state(checkpoint)
    if global_step != int(match.group(1)):
        raise ValueError("checkpoint directory and trainer_state.json disagree")
    return checkpoint, global_step


def _save_checkpoint(
    accelerator: Accelerator,
    output_dir: Path,
    global_step: int,
    keep_last: int,
) -> Path:
    checkpoint = output_dir / f"step-{global_step:08d}"
    accelerator.wait_for_everyone()
    accelerator.save_state(str(checkpoint))
    if accelerator.is_main_process:
        _write_state(checkpoint, global_step)
        for stale in _checkpoints(output_dir)[:-keep_last]:
            shutil.rmtree(stale)
    accelerator.wait_for_everyone()
    return checkpoint


def train_steps(
    cfg: Any,
    model: nn.Module,
    dataloader: Any,
    criterion: nn.Module,
) -> TrainingResult:
    """Train until ``max_steps`` and save Accelerate-compatible checkpoints."""

    try:
        if len(dataloader) == 0:
            raise ValueError("training dataloader is empty")
    except TypeError:
        pass

    train_cfg = _get(cfg, "train")
    scheduler_cfg = _get(cfg, "scheduler")
    checkpoint_cfg = _get(cfg, "checkpoint")
    logging_cfg = _get(cfg, "logging", {})
    max_steps = _positive_int("train.max_steps", _get(train_cfg, "max_steps"))
    accumulation = _positive_int(
        "train.gradient_accumulation_steps",
        _get(train_cfg, "gradient_accumulation_steps", 1),
    )
    save_every = _positive_int(
        "checkpoint.save_every_steps",
        _get(checkpoint_cfg, "save_every_steps"),
    )
    keep_last = _positive_int(
        "checkpoint.keep_last", _get(checkpoint_cfg, "keep_last", 1)
    )
    log_every = _positive_int(
        "logging.every_steps", _get(logging_cfg, "every_steps", 1)
    )
    max_grad_value = float(_get(train_cfg, "max_grad_value", 0.0))
    if max_grad_value < 0:
        raise ValueError("train.max_grad_value must be non-negative")

    output_dir = Path(str(_get(checkpoint_cfg, "output_dir"))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_from = _get(train_cfg, "resume_from")
    existing_checkpoints = _checkpoints(output_dir)
    global_step = 0
    resume_dir: Path | None = None
    if resume_from is None:
        if existing_checkpoints:
            raise ValueError(
                "fresh training requires checkpoint.output_dir without step checkpoints"
            )
    else:
        resume_dir, global_step = _resume_checkpoint(output_dir, resume_from)
        if existing_checkpoints[-1] != resume_dir:
            raise ValueError("train.resume_from must select the latest step checkpoint")
        if global_step > max_steps:
            raise ValueError("resume checkpoint is beyond train.max_steps")

    seed = int(_get(cfg, "seed", 83))
    backend = _get(logging_cfg, "backend")
    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation,
        mixed_precision=str(_get(train_cfg, "mixed_precision", "no")),
        log_with=backend,
        project_dir=str(output_dir),
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
            data_seed=seed,
            use_stateful_dataloader=True,
        ),
    )
    optimizer = build_optimizer(
        model,
        lr=_get(train_cfg, "lr"),
        weight_decay=_get(train_cfg, "weight_decay", 0.0),
        encoder_lr_scale=_get(train_cfg, "encoder_lr_scale", 0.2),
    )
    scheduler = StepLR(
        optimizer,
        step_size=_positive_int(
            "scheduler.step_size", _get(scheduler_cfg, "step_size")
        ),
        gamma=float(_get(scheduler_cfg, "gamma")),
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    criterion.to(accelerator.device)
    model.train()

    if backend:
        accelerator.init_trackers("PartSAM")

    if resume_dir is not None:
        accelerator.load_state(str(resume_dir))
    last_saved_step = global_step

    optimizer.zero_grad(set_to_none=True)
    pending_stats: torch.Tensor | None = None
    while global_step < max_steps:
        saw_batch = False
        for batch in dataloader:
            saw_batch = True
            if not isinstance(batch, Mapping):
                raise TypeError("training dataloader must return a mapping")
            coords, color, normal, gt_masks = _model_inputs(batch)
            target = _loss_target(gt_masks)

            with accelerator.accumulate(model):
                outputs = model(coords, color, normal, gt_masks, is_eval=False)
                loss, auxiliary = criterion(outputs, target, step=global_step)
                batch_stats = _log_statistics(loss, auxiliary)
                pending_stats = (
                    batch_stats if pending_stats is None else pending_stats + batch_stats
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients and max_grad_value > 0:
                    accelerator.clip_grad_value_(model.parameters(), max_grad_value)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            global_step += 1

            if global_step % log_every == 0 or global_step == max_steps:
                if pending_stats is None:
                    raise RuntimeError("missing training statistics")
                totals = accelerator.reduce(pending_stats, reduction="sum")
                metrics = {"loss": float((totals[0] / totals[1]).item())}
                if totals[3] > 0:
                    metrics["iou_at_1"] = float((totals[2] / totals[3]).item())
                    metrics["iou_at_final"] = float((totals[4] / totals[5]).item())
                accelerator.print(
                    f"step {global_step}/{max_steps} "
                    + " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
                )
                if backend:
                    accelerator.log(metrics, step=global_step)
                pending_stats = None

            if global_step % save_every == 0:
                _save_checkpoint(accelerator, output_dir, global_step, keep_last)
                last_saved_step = global_step
            if global_step >= max_steps:
                break

        if not saw_batch:
            raise ValueError("training dataloader is empty")

    if last_saved_step != global_step:
        _save_checkpoint(accelerator, output_dir, global_step, keep_last)
    if backend:
        accelerator.end_training()
    return TrainingResult(global_step, tuple(_checkpoints(output_dir)))


__all__ = ["TrainingResult", "build_optimizer", "train_steps"]
