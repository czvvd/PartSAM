import sys

sys.path.append(".")

import os
import argparse
import hydra
from omegaconf import OmegaConf
import matplotlib.colors as mcolors
import pointops
import torch
import numpy as np
import trimesh
import gc
import time
from collections import defaultdict,Counter, deque
from tqdm import tqdm
import networkx as nx
import igraph
import multiprocessing as mp


def construct_mesh_graph(mesh):
    mesh_edges = trimesh.graph.face_adjacency(mesh=mesh)
    mesh_graph = defaultdict(set)
    for face1, face2 in mesh_edges:
        mesh_graph[face1].add(face2)
        mesh_graph[face2].add(face1)
    return mesh_graph


def construct_mesh_graphMy(mesh):
    # 1. 获取原始邻接关系
    mesh_edges = trimesh.graph.face_adjacency(mesh=mesh)
    mesh_graph = defaultdict(set)
    for face1, face2 in mesh_edges:
        mesh_graph[face1].add(face2)
        mesh_graph[face2].add(face1)
    
    # 2. 检测连通分量
    visited = set()
    components = []
    for face in mesh_graph.keys():
        if face not in visited:
            # BFS遍历单个连通分量
            component = set()
            queue = deque([face])
            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    component.add(current)
                    # 添加未访问的邻居面
                    queue.extend(n for n in mesh_graph[current] if n not in visited)
            components.append(component)
    
    # 3. 若存在多个组件，添加虚拟边合并
    if len(components) > 1:
        for i in range(len(components) - 1):
            # 从相邻组件中各选一个代表面（如最小索引面）
            face_a = min(components[i])
            face_b = min(components[i+1])
            # 添加双向虚拟边
            mesh_graph[face_a].add(face_b)
            mesh_graph[face_b].add(face_a)
    
    return mesh_graph

def label_components(face2label, mesh, mesh_graph):
    """
    """
    components = []
    visited = set()

    def dfs(source: int):
        stack = [source]
        components.append({source})
        visited.add(source)
        
        while stack:
            node = stack.pop()
            for adj in mesh_graph[node]:
                if adj not in visited and adj in face2label and face2label[adj] == face2label[node]:
                    stack.append(adj)
                    components[-1].add(adj)
                    visited.add(adj)

    for face in range(len(mesh.faces)):
        if face not in visited and face in face2label:
            dfs(face)
    return components

def smooth(face2label_consistent,mesh,mesh_graph,components):
    """
    """
    # remove holes

    threshold_percentage_size = 0.005
    threshold_percentage_area = 0.005
    components = sorted(components, key=lambda x: len(x), reverse=True)
    components_area = [
        sum([float(mesh.area_faces[face]) for face in comp]) for comp in components
    ]
    max_size = max([len(comp) for comp in components])
    max_area = max(components_area)

    remove_comp_size = set()
    remove_comp_area = set()
    for i, comp in enumerate(components):
        if len(comp)          < max_size * threshold_percentage_size:
            remove_comp_size.add(i)
        if components_area[i] < max_area * threshold_percentage_area:
            remove_comp_area.add(i)
    remove_comp = remove_comp_size.intersection(remove_comp_area)
    print('Removing ', len(remove_comp), ' components')
    for i in remove_comp:
        for face in components[i]:
            face2label_consistent.pop(face)
    
    # fill islands
    print('Smoothing labels')
    smooth_iterations = 32
    for iteration in range(smooth_iterations):
        count = 0
        changes = {}
        for face in range(len(mesh.faces)):
            if face in face2label_consistent:
                continue
            labels_adj = Counter()
            for adj in mesh_graph[face]:
                if adj in face2label_consistent:
                    label = face2label_consistent[adj]
                    if label != 0:
                        labels_adj[label] += 1
            if len(labels_adj):
                count += 1
                changes[face] = labels_adj.most_common(1)[0][0]
        for face, label in changes.items():
            face2label_consistent[face] = label
        #print('Smoothing iteration ', iteration, ' changed ', count, ' faces')

    return face2label_consistent


def split(face2label_consistent, components):
    """
    """

    labels_seen = set()
    labels_curr = max(face2label_consistent.values()) + 1
    labels_orig = labels_curr
    for comp in components:
        face = comp.pop()
        label = face2label_consistent[face]
        comp.add(face)
        if label == 0 or label in labels_seen: # background or repeated label
            face2label_consistent.update({face: labels_curr for face in comp})
            labels_curr += 1
        labels_seen.add(label)
    print('Split', (labels_curr - labels_orig), 'times') # account for background

    return face2label_consistent

def construct_expansion_graph(
    label,
    mesh,
    partition,
    cost_data,
    cost_smoothness
) -> nx.Graph:
    """
    """
    G = nx.Graph() # undirected graph
    A = 'alpha'
    B = 'alpha_complement'

    node2index = {}
    G.add_node(A)
    G.add_node(B)
    node2index[A] = 0
    node2index[B] = 1
    for i in range(len(mesh.faces)):
        G.add_node(i)
        node2index[i] = 2 + i

    aux_count = 0
    for i, edge in enumerate(mesh.face_adjacency): # auxillary nodes
        f1, f2 = int(edge[0]), int(edge[1])
        if partition[f1] != partition[f2]:
            a = (f1, f2)
            if a in node2index: # duplicate edge
                continue
            G.add_node(a)
            node2index[a] = len(mesh.faces) + 2 + aux_count
            aux_count += 1

    for f in range(len(mesh.faces)):
        G.add_edge(A, f, capacity=cost_data[f, label])
        G.add_edge(B, f, capacity=float('inf') if partition[f] == label else cost_data[f, partition[f]])

    for i, edge in enumerate(mesh.face_adjacency):
        f1, f2 = int(edge[0]), int(edge[1])
        a = (f1, f2)
        if partition[f1] == partition[f2]:
            if partition[f1] != label:
                G.add_edge(f1, f2, capacity=cost_smoothness[i])
        else:
            G.add_edge(a, B, capacity=cost_smoothness[i])
            if partition[f1] != label:
                G.add_edge(f1, a, capacity=cost_smoothness[i])
            if partition[f2] != label:
                G.add_edge(a, f2, capacity=cost_smoothness[i])
    
    return G, node2index

def partition_cost(
    mesh,
    partition,
    cost_data,
    cost_smoothness
) -> float:
    """
    """
    cost = 0
    for f in range(len(partition)):
        cost += cost_data[f, partition[f]]
    for i, edge in enumerate(mesh.face_adjacency):
        f1, f2 = int(edge[0]), int(edge[1])
        if partition[f1] != partition[f2]:
            cost += cost_smoothness[i]
    return cost

def repartition(
    mesh: trimesh.Trimesh,
    partition,
    cost_data,
    cost_smoothness,
    smoothing_iterations,
    _lambda=1.0,
):
    A = 'alpha'
    B = 'alpha_complement'
    labels = np.unique(partition)

    cost_smoothness = cost_smoothness * _lambda

    # networkx broken for float capacities
    #cost_data       = np.round(cost_data       * SCALE).astype(int)
    #cost_smoothness = np.round(cost_smoothness * SCALE).astype(int)

    cost_min = partition_cost(mesh, partition, cost_data, cost_smoothness)

    for i in range(smoothing_iterations):

        #print('Repartition iteration ', i)
        
        for label in tqdm(labels):
            G, node2index = construct_expansion_graph(label, mesh, partition, cost_data, cost_smoothness)
            index2node = {v: k for k, v in node2index.items()}

            '''
            _, (S, T) = nx.minimum_cut(G, A, B)
            assert A in S and B in T
            S = np.array([v for v in S if isinstance(v, int)]).astype(int)
            T = np.array([v for v in T if isinstance(v, int)]).astype(int)
            '''

            G = igraph.Graph.from_networkx(G)
            outputs = G.st_mincut(source=node2index[A], target=node2index[B], capacity='capacity')
            S = outputs.partition[0]
            T = outputs.partition[1]
            assert node2index[A] in S and node2index[B] in T
            S = np.array([index2node[v] for v in S if isinstance(index2node[v], int)]).astype(int)
            T = np.array([index2node[v] for v in T if isinstance(index2node[v], int)]).astype(int)

            assert (partition[S] == label).sum() == 0 # T consists of those assigned 'alpha' and S 'alpha_complement' (see paper)
            partition[T] = label

            cost = partition_cost(mesh, partition, cost_data, cost_smoothness)
            if cost > cost_min:
                raise ValueError('Cost increased. This should not happen because the graph cut is optimal.')
            cost_min = cost
    
    return partition

def smooth_repartition_faces(face2label_consistent: dict, target_labels=None, mesh=None) -> dict:
    """
    """
    tmesh = mesh

    partition = np.array([face2label_consistent[face] for face in range(len(tmesh.faces))])

    cost_data = np.zeros((len(tmesh.faces), np.max(partition) + 1))
    for f in range(len(tmesh.faces)):
        for l in np.unique(partition):
            cost_data[f, l] = 0 if partition[f] == l else 1
    cost_smoothness = -np.log(tmesh.face_adjacency_angles / np.pi + 1e-20)
    
    lambda_seed = 6
    if target_labels is None:
        refined_partition = repartition(tmesh, partition, cost_data, cost_smoothness, 1, lambda_seed)
        return {
            face: refined_partition[face] for face in range(len(tmesh.faces))
        }

    lambda_range=(
        1, 
        15
    )
    lambdas = np.linspace(*lambda_range, num=mp.cpu_count())
    chunks = [
        (tmesh, partition, cost_data, cost_smoothness, 1, _lambda) 
        for _lambda in lambdas
    ]
    with mp.Pool(mp.cpu_count() // 2) as pool:
        refined_partitions = pool.starmap(repartition, chunks)

    def compute_cur_labels(part, noise_threshold=10):
        """
        """
        values, counts = np.unique(part, return_counts=True)
        return values[counts > noise_threshold]

    # lambda crawling algorithm when target_labels is specified i.e. Princeton Mesh Segmentation Benchmark
    max_iteration = 1
    cur_iteration = 0
    cur_lambda_index = np.searchsorted(lambdas, lambda_seed)
    cur_labels = len(compute_cur_labels(refined_partitions[cur_lambda_index]))
    while not (
        target_labels - 1 <= cur_labels and
        target_labels + 1 >= cur_labels
    ) and cur_iteration < max_iteration:
        
        if cur_labels < target_labels and cur_lambda_index > 0:
            # want more labels so decrease lambda
            cur_lambda_index -= 1
        if cur_labels > target_labels and cur_lambda_index < len(refined_partitions) - 1:
            # want less labels so increase lambda
            cur_lambda_index += 1

        cur_labels = len(compute_cur_labels(refined_partitions[cur_lambda_index]))
        cur_iteration += 1

    print('Repartitioned with ', cur_labels, ' labels aiming for ', target_labels, 'target labels using lambda ', lambdas[cur_lambda_index], ' in ', cur_iteration, ' iterations')
    
    refined_partition = refined_partitions[cur_lambda_index]
    return {
        face: refined_partition[face] for face in range(len(tmesh.faces))
    }

def post_processing1(mesh_group, mesh):
    face_labels = torch.from_numpy(mesh_group)
    label_dict = dict(zip(range(len(face_labels)), face_labels.tolist()))

    mesh_graph = construct_mesh_graphMy(mesh)
    components = label_components(label_dict,mesh,mesh_graph)
    label_smoothed = smooth(label_dict,mesh,mesh_graph,components)

    for face in range(len(mesh.faces)):
        if face not in label_smoothed:
            label_smoothed[face] = 0

    face2label_consistent = label_smoothed
    components = label_components(label_smoothed,mesh,mesh_graph)

    face2label_consistent = split(label_smoothed,components)

    face2label_consistent = smooth_repartition_faces(face2label_consistent, None, mesh)

    face2label_consistent = {int(k): int(v) for k, v in face2label_consistent.items()} # ensure serialization

    num_face = len(face2label_consistent)
    # 创建一个空的张量
    labels_tensor = torch.zeros(num_face, dtype=torch.int)
    # 填充张量
    for face_idx, label in face2label_consistent.items():
        labels_tensor[face_idx] = label
    mesh_group_new = labels_tensor.cpu().numpy()

    # Assign color to each face based on the label with most votes
    hex_colors = list(mcolors.CSS4_COLORS.values())
    rgb_colors = np.array([mcolors.to_rgb(color) for color in hex_colors if color not in ['#000000', '#FFFFFF']])
    def relative_luminance(color):
        return 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
    rgb_colors = [color for color in rgb_colors if (relative_luminance(color) > 0.4 and relative_luminance(color) < 0.8)]
    np.random.shuffle(rgb_colors)
    random_color = []
    for i in range(max(mesh_group_new) + 1):
        random_color.append(rgb_colors[i % len(rgb_colors)])
    random_color.append(np.array([0, 0, 0]))
    for face, label in enumerate(mesh_group_new):
        color = (random_color[label] * 255).astype(np.uint8)
        color_with_alpha = np.append(color, 255)  # Add alpha value
        mesh.visual.face_colors[face] = color_with_alpha
    return mesh