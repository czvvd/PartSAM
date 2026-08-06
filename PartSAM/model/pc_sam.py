"""Segment Anything Model for Point Clouds.

References:
- https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/sam.py

The interactive training forward is adapted from Point-SAM (BAAI-Vision), MIT;
see ``LICENSES/Point-SAM.txt``.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from .common import repeat_interleave, sample_interaction_prompts, sample_triplets
from .mask_decoder import AuxInputs, MaskDecoder, MLP
from .pc_encoder import PFEncoderDual
from .prompt_encoder import MaskEncoder, PointEncoder
from partfield.model.PVCNN.encoder_pc import sample_triplane_feat


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
        prompt_masks: torch.Tensor = None,
        multimask_output: bool = True,
    ):
        """Predict masks given point prompts.

        Args:
            coords: [B, N, 3]. Point cloud coordinates, normalized to [-1, 1].
            features: [B, N, F]. Point cloud features.
        """
        # pc_embeddings: [B, num_patches, D]
        if self.pc_embeddings is None:
            patches, pf_feat, part_planes = self.pc_encoder(coords, color, normal)
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

    def forward(
        self,
        coords: torch.Tensor,
        color: torch.Tensor,
        normal: torch.Tensor,
        gt_masks: torch.Tensor,
        *,
        is_eval: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """Run interactive training from one initial click through all rounds.

        Point-cloud and PartField features are computed once per shape.  Every
        round adds one click, and the decoder's highest predicted-IoU candidate
        becomes both the dense prompt and the error mask for the next round.
        """
        del is_eval  # Sampling follows the same deterministic paper rule in both modes.
        if self.prompt_iters < 1:
            raise ValueError("prompt_iters must be at least one")
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError(
                f"coords must have shape [B, N, 3], got {tuple(coords.shape)}"
            )
        if color.shape != coords.shape or normal.shape != coords.shape:
            raise ValueError(
                "color and normal must have the same shape as coords, got "
                f"{tuple(color.shape)} and {tuple(normal.shape)}"
            )
        if (
            gt_masks.ndim != 3
            or gt_masks.shape[0] != coords.shape[0]
            or gt_masks.shape[2] != coords.shape[1]
        ):
            raise ValueError(
                "gt_masks must have shape [B, M, N] matching coords, got "
                f"{tuple(gt_masks.shape)}"
            )
        if gt_masks.dtype != torch.bool:
            raise TypeError(f"gt_masks must be boolean, got {gt_masks.dtype}")

        batch_size = coords.shape[0]
        num_masks = gt_masks.shape[1]
        patches, pf_feat, part_planes = self.pc_encoder(coords, color, normal)
        pc_embeddings = patches["embeddings"]
        centers = patches["centers"]
        knn_idx = patches["knn_idx"]
        target_part_planes = repeat_interleave(part_planes, num_masks, dim=0)

        triplet_points = sample_triplets(coords, gt_masks)
        triplet_features = (
            sample_triplane_feat(target_part_planes, triplet_points)
            if triplet_points is not None
            else None
        )

        aux_inputs = AuxInputs(
            coords=coords,
            color=color,
            normal=normal,
            centers=centers,
        )
        pc_pe = self.point_encoder.pe_layer(centers)
        prompt_coords = coords.new_empty((batch_size * num_masks, 0, 3))
        prompt_labels = gt_masks.new_empty((batch_size * num_masks, 0))
        prompt_masks = None
        outputs = []

        for round_index in range(self.prompt_iters):
            new_prompt_coords, new_prompt_labels, _ = sample_interaction_prompts(
                coords, gt_masks, prompt_masks
            )
            prompt_coords = torch.cat([prompt_coords, new_prompt_coords], dim=1)
            prompt_labels = torch.cat([prompt_labels, new_prompt_labels], dim=1)

            prompt_partfield_features = sample_triplane_feat(
                target_part_planes, prompt_coords
            )
            sparse_embeddings = self.point_encoder(prompt_coords, prompt_labels)
            sparse_embeddings = sparse_embeddings + self.prompt_point_mapper(
                prompt_partfield_features
            )

            dense_embeddings = self.mask_encoder(
                prompt_masks,
                coords,
                centers,
                knn_idx,
                center_idx=patches.get("fps_idx"),
            )
            dense_embeddings = repeat_interleave(
                dense_embeddings,
                sparse_embeddings.shape[0] // dense_embeddings.shape[0],
                dim=0,
            )

            masks, iou_preds = self.mask_decoder(
                pc_embeddings,
                pc_pe,
                sparse_embeddings,
                dense_embeddings,
                aux_inputs=aux_inputs,
                multimask_output=round_index == 0,
                pf_feat=pf_feat,
            )
            prompt_masks_last = prompt_masks
            max_iou_pred_ind = torch.argmax(iou_preds, dim=1)
            prompt_masks = masks.gather(
                1,
                max_iou_pred_ind[:, None, None].expand(-1, 1, masks.shape[-1]),
            ).squeeze(1)

            outputs.append(
                {
                    "prompt_coords": prompt_coords,
                    "prompt_labels": prompt_labels,
                    "masks": masks,
                    "iou_preds": iou_preds,
                    "max_iou_pred_ind": max_iou_pred_ind,
                    "prompt_masks": prompt_masks,
                    "prompt_masks_last": prompt_masks_last,
                    "triplets": triplet_features,
                }
            )

        return outputs
