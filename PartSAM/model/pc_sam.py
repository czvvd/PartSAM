"""Segment Anything Model for Point Clouds.

References:
- https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/sam.py
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torkit3d.nn.functional import batch_index_select

from .common import repeat_interleave
from .mask_decoder import AuxInputs, MaskDecoder, MLP
from .pc_encoder import PFEncoderDual
from .prompt_encoder import MaskEncoder, PointEncoder
from partfield.model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat

  
class PartSAM(nn.Module):
    def __init__(
        self,
        pc_encoder: PFEncoderDual,
        mask_encoder: MaskEncoder,
        mask_decoder: MaskDecoder,
        prompt_iters: int,
        enable_mask_refinement_iterations=True,
    ):
        super().__init__()
        self.pc_encoder = pc_encoder
        self.mask_encoder = mask_encoder

        self.point_encoder = PointEncoder(mask_encoder.embed_dim)
        self.prompt_point_mapper = MLP(448, mask_encoder.embed_dim, mask_encoder.embed_dim, 2)

        self.mask_decoder = mask_decoder
        self.prompt_iters = prompt_iters
        self.enable_mask_refinement_iterations = enable_mask_refinement_iterations

        self.labels = None
        self.pc_embeddings = None
        self.patches = None
        self.pf_feat = None
        self.part_planes = None
 

    def predict_masks(
        self,
        coords: torch.Tensor,
        color: torch.Tensor,
        normal: torch.Tensor,
        point_to_face: torch.Tensor,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        prompt_coords: torch.Tensor,
        selected_indices: torch.Tensor,
        prompt_labels: torch.Tensor,
        mesh_cluster: torch.bool = False,
        prompt_masks: torch.Tensor = None,
        multimask_output: bool = True,
    ):
        """Predict masks given point prompts.

        Args:
            coords: [B, N, 3]. Point cloud coordinates, normalized to [-1, 1].
            features: [B, N, F]. Point cloud features.
        """
        # pc_embeddings: [B, num_patches, D]
        if self.pc_embeddings == None:
            patches, pf_feat, part_planes = self.pc_encoder(coords,color, normal)
            pc_embeddings = patches["embeddings"]
            self.pc_embeddings = pc_embeddings
            self.patches = patches
            self.pf_feat = pf_feat
            self.part_planes = part_planes
        else:
            pc_embeddings, patches, pf_feat, part_planes = self.pc_embeddings, self.patches, self.pf_feat, self.part_planes



        centers = patches["centers"]  # [B, num_patches, 3]
        knn_idx = patches["knn_idx"]  # [B, N, K]

        aux_inputs = AuxInputs(coords=coords, color=color, normal=normal, centers=centers)

        # [B, num_patches, D]
        pc_pe = self.point_encoder.pe_layer(centers)

        # Repeat part_planes along the batch dimension to match prompt_coords' first dimension
        repeat_times = prompt_coords.shape[0] // part_planes.shape[0]
        prompt_point_pffeat = sample_triplane_feat(part_planes.repeat(repeat_times, 1, 1, 1, 1), prompt_coords)

        # [B * M, num_queries, D]
        sparse_embeddings = self.point_encoder(prompt_coords, prompt_labels)
        sparse_embeddings =  sparse_embeddings + self.prompt_point_mapper(prompt_point_pffeat)
        
        # [B * M, num_patches, D] or [B, num_patches, D] (if prompt_masks=None)
        dense_embeddings = self.mask_encoder(
            prompt_masks,
            coords,
            centers,
            knn_idx,
            center_idx=patches.get("fps_idx"),
        )
        # [B * M, num_patches, D]
        dense_embeddings = repeat_interleave(
            dense_embeddings,
            sparse_embeddings.shape[0] // dense_embeddings.shape[0],
            0,
        )

        # [B * M, num_outputs, N], [B * M, num_outputs]
        masks, iou_preds = self.mask_decoder(
            pc_embeddings,
            pc_pe,
            sparse_embeddings,
            dense_embeddings,
            aux_inputs=aux_inputs,
            multimask_output=multimask_output,
            pf_feat = pf_feat
        )
        return masks, iou_preds
