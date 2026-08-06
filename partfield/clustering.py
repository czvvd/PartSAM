# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
#
# See LICENSES/NVIDIA-PartField.txt for the license distributed with this repository.

"""PartField's connectivity-constrained Ward clustering."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


CLUSTER_COUNTS = (10, 20, 30)


def _mesh_arrays(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must have non-empty shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("faces must have non-empty shape [F, 3]")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("faces must have an integer dtype")
    faces = faces.astype(np.int64, copy=False)
    if not np.isfinite(vertices).all():
        raise ValueError("vertices contain non-finite values")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("faces contain an invalid vertex index")
    return vertices, faces


def face_adjacency_edges(faces: np.ndarray) -> np.ndarray:
    """Return face pairs that share a complete mesh edge."""

    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("faces must have an integer dtype")

    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, (a, b, c) in enumerate(faces.tolist()):
        for left, right in ((a, b), (b, c), (c, a)):
            edge_faces[min(left, right), max(left, right)].append(face_index)

    pairs: set[tuple[int, int]] = set()
    for incident in edge_faces.values():
        pairs.update(combinations(sorted(set(incident)), 2))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(pairs), dtype=np.int64)


def l2_normalize_face_features(face_features: np.ndarray) -> np.ndarray:
    """L2-normalize every face feature, leaving zero vectors unchanged."""

    features = np.asarray(face_features, dtype=np.float64)
    if features.ndim != 2 or not features.shape[1]:
        raise ValueError("face_features must have shape [F, C]")
    if not np.isfinite(features).all():
        raise ValueError("face_features contain non-finite values")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)


def _knn_graph(points: np.ndarray, neighbors: int) -> csr_matrix:
    model = NearestNeighbors(n_neighbors=neighbors).fit(points)
    distances, indices = model.kneighbors(points)
    rows = np.repeat(np.arange(len(points)), neighbors)
    columns = indices.reshape(-1)
    weights = distances.reshape(-1)
    keep = rows != columns
    rows, columns = rows[keep], columns[keep]
    # Sparse graph routines treat exact zeros as missing edges.
    weights = np.maximum(weights[keep], 1e-12)
    graph = coo_matrix((weights, (rows, columns)), shape=(len(points),) * 2)
    graph = graph.maximum(graph.T).tocsr()
    return graph


def face_connectivity(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    k: int = 10,
    with_knn: bool = True,
) -> csr_matrix:
    """Build shared-edge adjacency plus PartField component-centroid bridges.

    For disconnected meshes, each component is represented by the face nearest
    its centroid.  A component-centroid KNN graph supplies the optional KNN
    edges and an MST supplies the minimum bridges needed for one connected
    graph.  The neighbor count is enlarged only if needed to guarantee that the
    MST itself is connected.
    """

    vertices, faces = _mesh_arrays(vertices, faces)
    if k <= 0:
        raise ValueError("k must be positive")
    face_count = len(faces)
    pairs = face_adjacency_edges(faces)
    if len(pairs):
        rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
        columns = np.concatenate((pairs[:, 1], pairs[:, 0]))
        adjacency = coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, columns)),
            shape=(face_count, face_count),
        ).tocsr()
    else:
        adjacency = csr_matrix((face_count, face_count), dtype=np.int8)

    component_count, component_labels = connected_components(
        adjacency, directed=False
    )
    if component_count == 1:
        return adjacency

    face_centers = vertices[faces].mean(axis=1)
    component_centers = np.empty((component_count, 3), dtype=np.float64)
    representatives = np.empty(component_count, dtype=np.int64)
    for component in range(component_count):
        members = np.flatnonzero(component_labels == component)
        center = face_centers[members].mean(axis=0)
        component_centers[component] = center
        representatives[component] = members[
            np.argmin(np.square(face_centers[members] - center).sum(axis=1))
        ]

    base_neighbors = min(int(k), component_count)
    knn = _knn_graph(component_centers, base_neighbors)
    mst_source = knn
    while connected_components(mst_source, directed=False, return_labels=False) > 1:
        next_neighbors = min(component_count, max(2, mst_source.getnnz(axis=1).max() * 2))
        if next_neighbors <= base_neighbors:
            next_neighbors = min(component_count, base_neighbors + 1)
        mst_source = _knn_graph(component_centers, int(next_neighbors))
        base_neighbors = int(next_neighbors)

    extra_pairs: set[tuple[int, int]] = set()
    tree = minimum_spanning_tree(mst_source)
    tree_rows, tree_columns = tree.nonzero()
    for left, right in zip(tree_rows.tolist(), tree_columns.tolist()):
        extra_pairs.add(tuple(sorted((int(representatives[left]), int(representatives[right])))))

    if with_knn:
        initial_knn = _knn_graph(component_centers, min(int(k), component_count))
        knn_rows, knn_columns = initial_knn.nonzero()
        for left, right in zip(knn_rows.tolist(), knn_columns.tolist()):
            extra_pairs.add(tuple(sorted((int(representatives[left]), int(representatives[right])))))

    extras = np.asarray(sorted(extra_pairs), dtype=np.int64)
    rows = np.concatenate((extras[:, 0], extras[:, 1]))
    columns = np.concatenate((extras[:, 1], extras[:, 0]))
    adjacency = adjacency + coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, columns)),
        shape=adjacency.shape,
    ).tocsr()
    adjacency.data[:] = 1
    if connected_components(adjacency, directed=False, return_labels=False) != 1:
        raise RuntimeError("could not connect mesh components")
    return adjacency


def cluster_face_features(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_features: np.ndarray,
    *,
    cluster_count: int,
) -> np.ndarray:
    """Run sklearn Ward clustering with PartField mesh connectivity."""

    vertices, faces = _mesh_arrays(vertices, faces)
    features = np.asarray(face_features)
    if features.ndim != 2 or features.shape[0] != len(faces):
        raise ValueError("face_features must have shape [F, C]")
    if isinstance(cluster_count, bool) or not isinstance(cluster_count, (int, np.integer)):
        raise ValueError("cluster_count must be an integer")
    cluster_count = int(cluster_count)
    if not 1 <= cluster_count <= len(faces):
        raise ValueError("cluster_count must be between 1 and the face count")
    if cluster_count == len(faces):
        return np.arange(len(faces), dtype=np.int32)

    labels = AgglomerativeClustering(
        n_clusters=cluster_count,
        linkage="ward",
        connectivity=face_connectivity(vertices, faces, k=10, with_knn=True),
        compute_full_tree=True,
    ).fit_predict(l2_normalize_face_features(features))

    # Relabel by first face occurrence so serialized labels are deterministic.
    unique = sorted(np.unique(labels).tolist(), key=lambda label: np.flatnonzero(labels == label)[0])
    dense = np.empty(len(labels), dtype=np.int32)
    for new_label, old_label in enumerate(unique):
        dense[labels == old_label] = new_label
    return dense


__all__ = [
    "CLUSTER_COUNTS",
    "cluster_face_features",
    "face_adjacency_edges",
    "face_connectivity",
    "l2_normalize_face_features",
]
