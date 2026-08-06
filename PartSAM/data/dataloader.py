"""Local GLB loading and part-mask datasets for PartSAM training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from torch.utils.data import Dataset


MIN_PARTS = 3
MAX_PARTS = 50
MIN_PSEUDO_MASKS = 6


class MeshRejected(RuntimeError):
    """A mesh that cannot provide the requested supervision."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class CanonicalMesh:
    mesh: trimesh.Trimesh
    face_part_ids: np.ndarray
    label_source: str
    raw_part_count: int


@dataclass(frozen=True)
class GeometrySample:
    coords: np.ndarray
    color: np.ndarray
    normal: np.ndarray
    point_to_face: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class PointSample:
    coords: np.ndarray
    color: np.ndarray
    normal: np.ndarray
    gt_masks: np.ndarray
    point_to_face: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray


def _validated_mesh(mesh: trimesh.Trimesh, detail: str) -> trimesh.Trimesh:
    mesh = mesh.copy()
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if len(vertices) == 0 or len(faces) == 0:
        raise MeshRejected("empty", detail)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise MeshRejected("non_triangular", detail)
    if not np.isfinite(vertices).all():
        raise MeshRejected("non_finite", detail)

    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    valid_faces = np.isfinite(areas) & (areas > 0)
    if not np.any(valid_faces):
        raise MeshRejected("zero_area", detail)
    if not np.all(valid_faces):
        mesh.update_faces(valid_faces)
        mesh.remove_unreferenced_vertices()
    return mesh


def _connected_components(mesh: trimesh.Trimesh) -> np.ndarray:
    """Label faces joined by a shared edge."""

    parent = np.arange(len(mesh.faces), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for first, second in np.asarray(mesh.face_adjacency, dtype=np.int64):
        first_root, second_root = find(int(first)), find(int(second))
        if first_root != second_root:
            parent[second_root] = first_root

    roots: dict[int, int] = {}
    labels = np.empty(len(mesh.faces), dtype=np.int64)
    for face_index in range(len(mesh.faces)):
        root = find(face_index)
        labels[face_index] = roots.setdefault(root, len(roots))
    return labels


def _normalize_mesh(mesh: trimesh.Trimesh) -> None:
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise MeshRejected("non_finite", "invalid mesh bounds")
    extent = float((bounds[1] - bounds[0]).max())
    if extent <= 0:
        raise MeshRejected("zero_area", "mesh has zero bounding-box extent")
    mesh.apply_translation(-bounds.mean(axis=0))
    mesh.apply_scale(1.8 / extent)


def canonicalize_scene(
    scene_or_mesh: trimesh.Scene | trimesh.Trimesh,
) -> CanonicalMesh:
    """Apply node transforms, merge the scene, and assign face-level parts."""

    if isinstance(scene_or_mesh, trimesh.Trimesh):
        scene = trimesh.Scene(scene_or_mesh)
    elif isinstance(scene_or_mesh, trimesh.Scene):
        scene = scene_or_mesh
    else:
        raise MeshRejected("empty", f"unsupported type: {type(scene_or_mesh).__name__}")

    nodes = list(scene.graph.nodes_geometry)
    if not nodes:
        raise MeshRejected("empty", "scene has no geometry nodes")

    instances: list[tuple[bytes, trimesh.Trimesh]] = []
    for node in nodes:
        transform, geometry_name = scene.graph.get(node)
        geometry = scene.geometry.get(geometry_name)
        if not isinstance(geometry, trimesh.Trimesh):
            raise MeshRejected("non_triangular", f"node {node!r} is not a mesh")
        mesh = _validated_mesh(geometry, f"node {node!r}")
        if not np.isfinite(transform).all():
            raise MeshRejected("non_finite", f"node {node!r} transform")
        mesh.apply_transform(transform)
        mesh = _validated_mesh(mesh, f"transformed node {node!r}")
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(mesh.vertices, dtype="<f8").tobytes())
        digest.update(np.ascontiguousarray(mesh.faces, dtype="<i8").tobytes())
        instances.append((digest.digest(), mesh))

    # Trimesh adds random suffixes when a GLB contains duplicate node names.
    # Sorting by transformed geometry keeps face order stable across reloads.
    instances.sort(key=lambda item: item[0])
    meshes = [mesh for _, mesh in instances]

    if len(meshes) == 1:
        mesh = meshes[0]
        face_part_ids = _connected_components(mesh)
        label_source = "connected_components"
    else:
        face_part_ids = np.concatenate(
            [np.full(len(mesh.faces), index, dtype=np.int64) for index, mesh in enumerate(meshes)]
        )
        mesh = trimesh.util.concatenate(meshes)
        label_source = "scene_nodes"

    _normalize_mesh(mesh)
    return CanonicalMesh(
        mesh=mesh,
        face_part_ids=face_part_ids,
        label_source=label_source,
        raw_part_count=int(face_part_ids.max()) + 1,
    )


def load_canonical_glb(path: str | Path) -> CanonicalMesh:
    """Load one GLB without trimesh processing and canonicalize it."""

    try:
        loaded = trimesh.load(path, force="scene", process=False)
    except Exception as error:
        raise MeshRejected("load_failed", f"{type(error).__name__}: {error}") from error
    return canonicalize_scene(loaded)


def inspect_mesh(path: str | Path) -> CanonicalMesh:
    """Compatibility alias used by the preparation and curation scripts."""

    return load_canonical_glb(path)


def _sample_surface(mesh: trimesh.Trimesh, count: int, seed: int):
    try:
        sampler = importlib.import_module("utils.point").sample_surface
        return sampler(mesh, count, sample_color=True, seed=seed)
    except Exception:
        try:
            sampler = importlib.import_module("utils.point").sample_surface
            coords, face_ids = sampler(mesh, count, sample_color=False, seed=seed)
            return coords, face_ids, None
        except Exception as error:
            raise MeshRejected("sampling_failed", str(error)) from error


def _sample_colors(mesh: trimesh.Trimesh, colors: Any, count: int) -> np.ndarray:
    if not bool(mesh.visual.defined) or colors is None:
        return np.full((count, 3), 192, dtype=np.uint8)
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[0] != count or colors.shape[1] < 3:
        return np.full((count, 3), 192, dtype=np.uint8)
    try:
        if not np.isfinite(colors[:, :3]).all():
            raise ValueError
        return np.clip(colors[:, :3], 0, 255).astype(np.uint8)
    except (TypeError, ValueError):
        return np.full((count, 3), 192, dtype=np.uint8)


def _sampling_options(
    num_points: int, min_part_ratio: float, max_part_ratio: float
) -> tuple[int, float, float]:
    num_points = int(num_points)
    minimum, maximum = float(min_part_ratio), float(max_part_ratio)
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if not 0 <= minimum < maximum <= 1:
        raise ValueError("part ratios must satisfy 0 <= min < max <= 1")
    return num_points, minimum, maximum


def sample_geometry(
    canonical: CanonicalMesh, *, num_points: int, seed: int
) -> GeometrySample:
    """Sample normalized geometry without assuming native part labels."""

    num_points = int(num_points)
    seed = int(seed)
    if num_points <= 0 or seed < 0:
        raise ValueError("num_points must be positive and seed non-negative")
    coords, point_to_face, colors = _sample_surface(canonical.mesh, num_points, seed)
    coords = np.asarray(coords, dtype=np.float32)
    point_to_face = np.asarray(point_to_face, dtype=np.int64)
    face_count = len(canonical.mesh.faces)
    if coords.shape != (num_points, 3) or not np.isfinite(coords).all():
        raise MeshRejected("sampling_failed", "invalid sampled coordinates")
    if (
        point_to_face.shape != (num_points,)
        or np.any(point_to_face < 0)
        or np.any(point_to_face >= face_count)
    ):
        raise MeshRejected("sampling_failed", "invalid sampled face indices")

    normals = np.asarray(canonical.mesh.face_normals, dtype=np.float64)[point_to_face]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    return GeometrySample(
        coords=coords,
        color=_sample_colors(canonical.mesh, colors, num_points),
        normal=normals.astype(np.float32),
        point_to_face=point_to_face,
        vertices=np.asarray(canonical.mesh.vertices, dtype=np.float32),
        faces=np.asarray(canonical.mesh.faces, dtype=np.int64),
    )


def build_point_sample(
    canonical: CanonicalMesh,
    *,
    num_points: int,
    min_part_ratio: float = 0.003,
    max_part_ratio: float = 0.70,
    seed: int = 147,
) -> PointSample:
    """Sample a mesh and retain native parts inside the configured size range."""

    num_points, minimum, maximum = _sampling_options(
        num_points, min_part_ratio, max_part_ratio
    )
    face_part_ids = np.asarray(canonical.face_part_ids, dtype=np.int64)
    if face_part_ids.shape != (len(canonical.mesh.faces),):
        raise MeshRejected("invalid_labels", "face labels do not match the mesh")

    sample = sample_geometry(canonical, num_points=num_points, seed=seed)
    point_parts = face_part_ids[sample.point_to_face]
    ratios = np.bincount(
        point_parts, minlength=canonical.raw_part_count
    ).astype(np.float64) / num_points
    valid_ids = np.flatnonzero((ratios > minimum) & (ratios < maximum))
    if len(valid_ids) < MIN_PARTS:
        raise MeshRejected(
            "insufficient_parts", f"found {len(valid_ids)} valid parts; minimum is {MIN_PARTS}"
        )
    if len(valid_ids) > MAX_PARTS:
        raise MeshRejected(
            "too_many_parts", f"found {len(valid_ids)} valid parts; maximum is {MAX_PARTS}"
        )

    return PointSample(
        coords=sample.coords,
        color=sample.color,
        normal=sample.normal,
        gt_masks=(valid_ids[:, None] == point_parts[None, :]),
        point_to_face=sample.point_to_face,
        vertices=sample.vertices,
        faces=sample.faces,
    )


def iter_glbs(data_root: str | Path) -> list[tuple[str, Path]]:
    """Return local GLBs in stable relative-path order."""

    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")
    files = sorted(
        (
            (path.relative_to(root).as_posix(), path)
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".glb"
        ),
        key=lambda item: item[0],
    )
    if not files:
        raise ValueError(f"no GLB files found under {root}")
    return files


def sampling_seed(base_seed: int, source_path: str) -> int:
    """Derive a stable per-shape seed from its relative path."""

    base_seed = int(base_seed)
    if base_seed < 0:
        raise ValueError("seed must be non-negative")
    words = np.frombuffer(
        hashlib.sha256(source_path.encode("utf-8")).digest()[:16], dtype="<u4"
    )
    sequence = np.random.SeedSequence([base_seed, *(int(word) for word in words)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


class MeshDataset(Dataset):
    """Load GLB meshes supervised by their native mesh parts."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        num_points: int = 100_000,
        min_part_ratio: float = 0.003,
        max_part_ratio: float = 0.70,
        seed: int = 83,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.num_points, self.min_part_ratio, self.max_part_ratio = _sampling_options(
            num_points, min_part_ratio, max_part_ratio
        )
        self.records: list[tuple[str, int]] = []
        for source_path, source in iter_glbs(self.data_root):
            shape_seed = sampling_seed(seed, source_path)
            try:
                build_point_sample(
                    load_canonical_glb(source),
                    num_points=self.num_points,
                    min_part_ratio=self.min_part_ratio,
                    max_part_ratio=self.max_part_ratio,
                    seed=shape_seed,
                )
                self.records.append((source_path, shape_seed))
            except MeshRejected:
                continue
        if not self.records:
            raise ValueError("no GLB provides 3--50 valid native parts")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_path, shape_seed = self.records[index]
        canonical = load_canonical_glb(self.data_root / source_path)
        sample = build_point_sample(
            canonical,
            num_points=self.num_points,
            min_part_ratio=self.min_part_ratio,
            max_part_ratio=self.max_part_ratio,
            seed=shape_seed,
        )
        return {
            "coords": sample.coords,
            "color": sample.color,
            "normal": sample.normal,
            "gt_masks": sample.gt_masks,
            "source_path": source_path,
        }


class PseudoLabelDataset(Dataset):
    """Load boolean face masks produced by the PartField curation script."""

    def __init__(
        self,
        data_root: str | Path,
        label_root: str | Path,
        *,
        num_points: int = 100_000,
        min_part_ratio: float = 0.003,
        max_part_ratio: float = 0.70,
        seed: int = 83,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.label_root = Path(label_root).expanduser().resolve()
        self.num_points, self.min_part_ratio, self.max_part_ratio = _sampling_options(
            num_points, min_part_ratio, max_part_ratio
        )
        if not self.label_root.is_dir():
            raise FileNotFoundError(f"pseudo-label root does not exist: {self.label_root}")

        meshes_by_uid: dict[str, tuple[str, Path]] = {}
        for source_path, source in iter_glbs(self.data_root):
            uid = source.stem
            if uid in meshes_by_uid:
                raise ValueError(f"duplicate mesh ID: {uid}")
            meshes_by_uid[uid] = (source_path, source)

        self.records: list[tuple[str, Path, int]] = []
        for label_path in sorted(self.label_root.glob("*.npy")):
            record = meshes_by_uid.get(label_path.stem)
            if record is None:
                raise FileNotFoundError(
                    f"no GLB matches pseudo label {label_path.name}"
                )
            source_path, _ = record
            self.records.append(
                (source_path, label_path, sampling_seed(seed, source_path))
            )
        if not self.records:
            raise ValueError("pseudo-label root contains no .npy files")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source_path, label_path, shape_seed = self.records[index]
        canonical = load_canonical_glb(self.data_root / source_path)
        face_masks = np.load(label_path, allow_pickle=False)
        if face_masks.dtype != np.bool_ or face_masks.ndim != 2:
            raise ValueError(f"{label_path.name} must contain a boolean [M, F] array")
        if face_masks.shape[1] != len(canonical.mesh.faces):
            raise ValueError(f"{label_path.name} does not match the source mesh")

        sample = sample_geometry(
            canonical, num_points=self.num_points, seed=shape_seed
        )
        point_masks = face_masks[:, sample.point_to_face]
        ratios = point_masks.mean(axis=1)
        valid = (ratios > self.min_part_ratio) & (ratios < self.max_part_ratio)
        point_masks = point_masks[valid]
        if len(point_masks) < MIN_PSEUDO_MASKS:
            raise MeshRejected(
                "insufficient_parts",
                f"found {len(point_masks)} valid pseudo masks; minimum is {MIN_PSEUDO_MASKS}",
            )
        return {
            "coords": sample.coords,
            "color": sample.color,
            "normal": sample.normal,
            "gt_masks": point_masks,
            "source_path": source_path,
        }


__all__ = [
    "CanonicalMesh",
    "GeometrySample",
    "MAX_PARTS",
    "MIN_PARTS",
    "MIN_PSEUDO_MASKS",
    "MeshRejected",
    "MeshDataset",
    "PseudoLabelDataset",
    "PointSample",
    "build_point_sample",
    "canonicalize_scene",
    "inspect_mesh",
    "iter_glbs",
    "load_canonical_glb",
    "sample_geometry",
    "sampling_seed",
]
