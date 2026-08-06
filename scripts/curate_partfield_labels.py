#!/usr/bin/env python3
"""Generate PartField pseudo labels and filter them with a trained PartSAM."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PartSAM.data.dataloader import (
    GeometrySample,
    MeshRejected,
    build_point_sample,
    iter_glbs,
    load_canonical_glb,
    sample_geometry,
    sampling_seed,
)
from partfield.clustering import CLUSTER_COUNTS, cluster_face_features


MASK_IOU_AT_1_THRESHOLD = 0.60
MASK_IOU_AT_10_THRESHOLD = 0.90
MIN_ACCEPTED_MASKS = 6
INTERACTION_ROUNDS = 10
MIN_PART_RATIO = 0.003
MAX_PART_RATIO = 0.70
FACE_SAMPLES = 10


def _iou(value: float, name: str) -> np.float32:
    value = np.float32(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def accept_mask(iou_at_1: float, iou_at_10: float) -> bool:
    """Paper rule: IoU@1 > 0.60 or IoU@10 > 0.90 (strict)."""

    return bool(
        _iou(iou_at_1, "IoU@1") > np.float32(MASK_IOU_AT_1_THRESHOLD)
        or _iou(iou_at_10, "IoU@10") > np.float32(MASK_IOU_AT_10_THRESHOLD)
    )


def accept_shape(valid_mask_count: int) -> bool:
    """A shape is useful only when at least six candidate masks pass."""

    if isinstance(valid_mask_count, bool) or not isinstance(
        valid_mask_count, (int, np.integer)
    ):
        raise ValueError("mask count must be an integer")
    if valid_mask_count < 0:
        raise ValueError("mask count must be non-negative")
    return int(valid_mask_count) >= MIN_ACCEPTED_MASKS


class MaskEvaluator(Protocol):
    def evaluate_masks(
        self, sample: GeometrySample, masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class FaceCurationResult:
    accepted: bool
    valid_mask_count: int
    accepted_ids: dict[int, np.ndarray]
    iou_at_1: dict[int, np.ndarray]
    iou_at_10: dict[int, np.ndarray]


def curate_face_candidates(
    *,
    face_labels: Mapping[int, np.ndarray],
    point_to_face: np.ndarray,
    evaluator: MaskEvaluator,
    sample: GeometrySample,
    min_part_ratio: float = MIN_PART_RATIO,
    max_part_ratio: float = MAX_PART_RATIO,
) -> FaceCurationResult:
    """Filter K=10/20/30 face clusters with size and PartSAM IoU rules."""

    if set(face_labels) != set(CLUSTER_COUNTS):
        raise ValueError(f"face_labels must contain {CLUSTER_COUNTS}")
    point_to_face = np.asarray(point_to_face, dtype=np.int64)
    if point_to_face.ndim != 1 or not len(point_to_face):
        raise ValueError("point_to_face must be a non-empty vector")
    if not 0 <= min_part_ratio < max_part_ratio <= 1:
        raise ValueError("part ratios must satisfy 0 <= min < max <= 1")

    labels_by_count: dict[int, np.ndarray] = {}
    face_count: int | None = None
    for count in CLUSTER_COUNTS:
        labels = np.asarray(face_labels[count], dtype=np.int32)
        if labels.ndim != 1:
            raise ValueError(f"K={count} labels must be one-dimensional")
        if face_count is None:
            face_count = len(labels)
        if len(labels) != face_count or np.unique(labels).size != count:
            raise ValueError(f"K={count} labels must contain exactly {count} parts")
        labels_by_count[count] = labels

    assert face_count is not None
    if np.any(point_to_face < 0) or np.any(point_to_face >= face_count):
        raise ValueError("point_to_face contains an invalid face index")

    masks: list[np.ndarray] = []
    keys: list[tuple[int, int]] = []
    for count in CLUSTER_COUNTS:
        point_labels = labels_by_count[count][point_to_face]
        for cluster_id in range(count):
            mask = point_labels == cluster_id
            ratio = float(mask.mean())
            if min_part_ratio < ratio < max_part_ratio:
                masks.append(mask)
                keys.append((count, cluster_id))

    if masks:
        iou_at_1, iou_at_10 = evaluator.evaluate_masks(
            sample, np.stack(masks).astype(np.bool_, copy=False)
        )
    else:
        iou_at_1 = iou_at_10 = np.empty(0, dtype=np.float32)
    iou_at_1 = np.asarray(iou_at_1, dtype=np.float32)
    iou_at_10 = np.asarray(iou_at_10, dtype=np.float32)
    if iou_at_1.shape != (len(keys),) or iou_at_10.shape != (len(keys),):
        raise ValueError("evaluator must return one IoU pair per candidate mask")

    ids = {count: [] for count in CLUSTER_COUNTS}
    first = {count: [] for count in CLUSTER_COUNTS}
    final = {count: [] for count in CLUSTER_COUNTS}
    for index, (count, cluster_id) in enumerate(keys):
        if accept_mask(iou_at_1[index], iou_at_10[index]):
            ids[count].append(cluster_id)
            first[count].append(iou_at_1[index])
            final[count].append(iou_at_10[index])

    accepted_ids = {
        count: np.asarray(ids[count], dtype=np.int32) for count in CLUSTER_COUNTS
    }
    accepted_first = {
        count: np.asarray(first[count], dtype=np.float32) for count in CLUSTER_COUNTS
    }
    accepted_final = {
        count: np.asarray(final[count], dtype=np.float32) for count in CLUSTER_COUNTS
    }
    valid_count = sum(len(values) for values in accepted_ids.values())
    return FaceCurationResult(
        accepted=accept_shape(valid_count),
        valid_mask_count=valid_count,
        accepted_ids=accepted_ids,
        iou_at_1=accepted_first,
        iou_at_10=accepted_final,
    )


def _normalize_for_model(sample: GeometrySample) -> GeometrySample:
    coords = np.asarray(sample.coords, dtype=np.float32).copy()
    vertices = np.asarray(sample.vertices, dtype=np.float32).copy()
    lower, upper = coords.min(axis=0), coords.max(axis=0)
    extent = float((upper - lower).max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("surface sample has zero extent")
    center = (lower + upper) * 0.5
    scale = 1.8 / extent
    return GeometrySample(
        coords=(coords - center) * scale,
        color=np.asarray(sample.color, dtype=np.float32) / 255.0,
        normal=np.asarray(sample.normal, dtype=np.float32),
        point_to_face=np.asarray(sample.point_to_face, dtype=np.int64),
        vertices=(vertices - center) * scale,
        faces=np.asarray(sample.faces, dtype=np.int64),
    )


def _safe_source(data_root: Path, source_path: str) -> Path:
    relative = PurePosixPath(source_path.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".glb":
        raise ValueError(f"invalid relative GLB path: {source_path!r}")
    path = (data_root / Path(*relative.parts)).resolve()
    path.relative_to(data_root)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_stage1_model(model_config: Path, checkpoint: Path) -> Any:
    if checkpoint.suffix.lower() != ".safetensors":
        raise ValueError("checkpoint must be model.safetensors")
    if not model_config.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("model config or checkpoint does not exist")
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    model = instantiate(OmegaConf.load(model_config))
    model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    model.prompt_iters = INTERACTION_ROUNDS
    return model


class PartSAMCurationBackend:
    """PartField feature extraction plus ten-round PartSAM evaluation."""

    def __init__(
        self,
        model: Any,
        *,
        device: str,
        batch_size: int = 4,
        feature_chunk_size: int = 65_536,
    ) -> None:
        import torch

        if batch_size <= 0 or feature_chunk_size < FACE_SAMPLES:
            raise ValueError("batch and feature chunk sizes must be positive")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.batch_size = int(batch_size)
        self.feature_chunk_size = int(feature_chunk_size)

    @classmethod
    def from_stage1_checkpoint(
        cls,
        *,
        model_config: str | Path,
        stage1_checkpoint: str | Path,
        device: str,
        batch_size: int = 4,
        feature_chunk_size: int = 65_536,
    ) -> "PartSAMCurationBackend":
        model = _load_stage1_model(
            Path(model_config).expanduser().resolve(),
            Path(stage1_checkpoint).expanduser().resolve(),
        )
        return cls(
            model,
            device=device,
            batch_size=batch_size,
            feature_chunk_size=feature_chunk_size,
        )

    def extract_face_features(self, sample: GeometrySample, *, seed: int) -> np.ndarray:
        """Average PartField features at ten random barycentric points per face."""

        import torch
        from partfield.model.PVCNN.encoder_pc import sample_triplane_feat

        coords = torch.as_tensor(sample.coords, dtype=torch.float32, device=self.device)[None]
        triangles = np.asarray(sample.vertices, dtype=np.float32)[sample.faces]
        rng = np.random.default_rng(seed)
        faces_per_chunk = max(1, self.feature_chunk_size // FACE_SAMPLES)
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            planes = self.model.pc_encoder.partfield(coords)
            if planes.ndim != 5 or planes.shape[2] <= 64:
                raise RuntimeError("PartField returned invalid triplanes")
            part_planes = planes[:, :, 64:]
            for start in range(0, len(triangles), faces_per_chunk):
                current = triangles[start : start + faces_per_chunk]
                random = rng.random((len(current), FACE_SAMPLES, 2))
                root = np.sqrt(random[..., 0])
                weights = np.stack(
                    (1.0 - root, root * (1.0 - random[..., 1]), root * random[..., 1]),
                    axis=-1,
                )
                points = np.einsum("fsv,fvd->fsd", weights, current).reshape(1, -1, 3)
                query = torch.as_tensor(points, dtype=torch.float32, device=self.device)
                features = sample_triplane_feat(part_planes, query)[0]
                chunks.append(
                    features.reshape(len(current), FACE_SAMPLES, -1)
                    .mean(dim=1)
                    .float()
                    .cpu()
                    .numpy()
                )
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _selected_logits(output: Mapping[str, Any]) -> Any:
        import torch

        masks = output["masks"]
        indices = output.get("max_iou_pred_ind")
        if indices is None:
            indices = torch.argmax(output["iou_preds"], dim=1)
        batch = torch.arange(masks.shape[0], device=masks.device)
        return masks[batch, indices.long()]

    def evaluate_masks(
        self, sample: GeometrySample, masks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        masks = np.asarray(masks, dtype=np.bool_)
        if masks.ndim != 2 or masks.shape[1] != len(sample.coords):
            raise ValueError("candidate masks must have shape [M, N]")
        coords = torch.as_tensor(sample.coords, dtype=torch.float32, device=self.device)[None]
        color = torch.as_tensor(sample.color, dtype=torch.float32, device=self.device)[None]
        normal = torch.as_tensor(sample.normal, dtype=torch.float32, device=self.device)[None]
        first, final = [], []
        with torch.inference_mode():
            for start in range(0, len(masks), self.batch_size):
                target = torch.as_tensor(
                    masks[start : start + self.batch_size],
                    dtype=torch.bool,
                    device=self.device,
                )
                outputs = self.model(coords, color, normal, target[None], is_eval=True)
                if len(outputs) != INTERACTION_ROUNDS:
                    raise RuntimeError("PartSAM must return ten interaction rounds")
                for destination, output in ((first, outputs[0]), (final, outputs[-1])):
                    prediction = self._selected_logits(output) > 0
                    intersection = (prediction & target).sum(dim=1).float()
                    union = (prediction | target).sum(dim=1).float()
                    destination.append((intersection / union.clamp_min(1)).cpu().numpy())
        return (
            np.concatenate(first).astype(np.float32, copy=False),
            np.concatenate(final).astype(np.float32, copy=False),
        )


@dataclass
class CurationSummary:
    accepted: int = 0
    rejected: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "failed": self.failed,
        }


def discover_stage2_candidates(
    data_root: str | Path,
    *,
    num_points: int,
    min_part_ratio: float,
    max_part_ratio: float,
    seed: int,
) -> list[str]:
    """Find GLBs with more than 50 valid native parts."""

    candidates: list[str] = []
    for source_path, source in iter_glbs(data_root):
        try:
            build_point_sample(
                load_canonical_glb(source),
                num_points=num_points,
                min_part_ratio=min_part_ratio,
                max_part_ratio=max_part_ratio,
                seed=sampling_seed(seed, source_path),
            )
        except MeshRejected as error:
            if error.reason == "too_many_parts":
                candidates.append(source_path)
    return candidates


def run_curation(
    *,
    data_root: str | Path,
    candidate_paths: Sequence[str],
    output_root: str | Path,
    backend: PartSAMCurationBackend,
    num_points: int,
    min_part_ratio: float,
    max_part_ratio: float,
    seed: int,
) -> CurationSummary:
    """Curate candidates and write accepted boolean face masks."""

    data_root = Path(data_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary = CurationSummary()

    for source_path in candidate_paths:
        try:
            source = _safe_source(data_root, source_path)
            canonical = load_canonical_glb(source)
            faces = np.asarray(canonical.mesh.faces)
            if len(faces) < max(CLUSTER_COUNTS):
                raise ValueError("mesh has fewer than 30 faces")
            shape_seed = sampling_seed(seed, source_path)
            sample = _normalize_for_model(
                sample_geometry(canonical, num_points=num_points, seed=shape_seed)
            )
            feature_seed = int(
                np.random.SeedSequence([shape_seed, 0x50465254]).generate_state(1)[0]
            )
            features = backend.extract_face_features(sample, seed=feature_seed)
            face_labels = {
                count: cluster_face_features(
                    sample.vertices,
                    sample.faces,
                    features,
                    cluster_count=count,
                )
                for count in CLUSTER_COUNTS
            }
            result = curate_face_candidates(
                face_labels=face_labels,
                point_to_face=sample.point_to_face,
                evaluator=backend,
                sample=sample,
                min_part_ratio=min_part_ratio,
                max_part_ratio=max_part_ratio,
            )
            if not result.accepted:
                summary.rejected += 1
                continue

            accepted_masks = [
                face_labels[count][None, :] == result.accepted_ids[count][:, None]
                for count in CLUSTER_COUNTS
                if len(result.accepted_ids[count])
            ]
            face_masks = np.concatenate(accepted_masks, axis=0).astype(
                np.bool_, copy=False
            )
            label_path = output_root / f"{Path(source_path).stem}.npy"
            if label_path.exists():
                raise FileExistsError(label_path)
            np.save(label_path, face_masks, allow_pickle=False)
            summary.accepted += 1
        except Exception as error:
            summary.failed += 1
            print(f"failed {source_path}: {type(error).__name__}: {error}", file=sys.stderr)

    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/objaverse"))
    parser.add_argument(
        "--checkpoint", type=Path, default=root / "pretrained/model.safetensors"
    )
    parser.add_argument(
        "--model-config", type=Path, default=root / "configs/model/PartSAM.yaml"
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/partfield_labels"))
    parser.add_argument("--num-points", type=int, default=100_000)
    parser.add_argument("--min-part-ratio", type=float, default=MIN_PART_RATIO)
    parser.add_argument("--max-part-ratio", type=float, default=MAX_PART_RATIO)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--feature-chunk-size", type=int, default=65_536)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.num_points <= 0 or args.seed < 0:
        raise ValueError("num-points must be positive and seed non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    candidates = discover_stage2_candidates(
        args.data_root,
        num_points=args.num_points,
        min_part_ratio=args.min_part_ratio,
        max_part_ratio=args.max_part_ratio,
        seed=args.seed,
    )
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        raise ValueError("no candidate GLBs found under data-root")

    backend = PartSAMCurationBackend.from_stage1_checkpoint(
        model_config=args.model_config,
        stage1_checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        feature_chunk_size=args.feature_chunk_size,
    )
    summary = run_curation(
        data_root=args.data_root,
        candidate_paths=candidates,
        output_root=args.output_root,
        backend=backend,
        num_points=args.num_points,
        min_part_ratio=args.min_part_ratio,
        max_part_ratio=args.max_part_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary.as_dict(), sort_keys=True))
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLUSTER_COUNTS",
    "FaceCurationResult",
    "PartSAMCurationBackend",
    "accept_mask",
    "accept_shape",
    "build_argument_parser",
    "curate_face_candidates",
    "discover_stage2_candidates",
    "main",
    "run_curation",
]
