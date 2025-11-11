# https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py
import dataclasses
from typing import Dict, List, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from .common import compute_interp_weights, interpolate_features, repeat_interleave



@dataclasses.dataclass
class AuxInputs:
    coords: torch.Tensor
    color: torch.Tensor
    normal: torch.Tensor
    centers: torch.Tensor
    interp_index: torch.Tensor = None
    interp_weight: torch.Tensor = None

@dataclasses.dataclass
class AuxInputs_ori:
    coords: torch.Tensor
    features: torch.Tensor
    centers: torch.Tensor
    interp_index: torch.Tensor = None
    interp_weight: torch.Tensor = None


class MaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        embedding_input_dim: int = 448,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim, 3)
                for i in range(self.num_mask_tokens)
            ]
        )
        self.encoder_mapper = MLP(embedding_input_dim, transformer_dim, transformer_dim, 3)
        self.output_upscaling = nn.Sequential(
            nn.Linear(transformer_dim+448, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        pc_embeddings: torch.Tensor,
        pc_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        aux_inputs: AuxInputs,
        multimask_output: bool,
        pf_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        pc_embeddings = self.encoder_mapper(pc_embeddings)

        masks, iou_pred = self.predict_masks(
            pc_embeddings=pc_embeddings,
            pf_feat = pf_feat,
            pc_pe=pc_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            aux_inputs=aux_inputs,
            mask_slice=mask_slice,
        )


        return masks, iou_pred

    def predict_masks(
        self,
        pc_embeddings: torch.Tensor,
        pf_feat: torch.Tensor,
        pc_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        aux_inputs: AuxInputs,
        mask_slice: slice = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        repeats = tokens.shape[0] // pc_embeddings.shape[0]
        src = repeat_interleave(pc_embeddings, repeats, dim=0)
        pos_src = repeat_interleave(pc_pe, repeats, dim=0)
        src = src + dense_prompt_embeddings

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        coords = aux_inputs.coords  # [B, N, 3]
        centers = aux_inputs.centers  # [B, L, 3]
        interp_index = aux_inputs.interp_index  # [B, N, 3]
        interp_weight = aux_inputs.interp_weight  # [B, N, 3]
        if interp_index is None or interp_weight is None:
            with torch.no_grad():
                interp_index, interp_weight = compute_interp_weights(coords, centers)
            aux_inputs.interp_index = interp_index
            aux_inputs.interp_weight = interp_weight

        _repeats = tokens.shape[0] // interp_index.shape[0]
        interp_index = repeat_interleave(interp_index, _repeats, dim=0)
        interp_weight = repeat_interleave(interp_weight, _repeats, dim=0)

        interp_embedding = interpolate_features(src, interp_index, interp_weight)
        if pf_feat.size(0) != interp_embedding.size(0):
            pf_feat = repeat_interleave(pf_feat, _repeats, dim=0)
        interp_embedding = torch.cat(
            (interp_embedding, pf_feat), dim=-1
        )
        upscaled_embedding = self.output_upscaling(interp_embedding)

        hyper_in_list: List[torch.Tensor] = []
        mask_indices = list(range(self.num_mask_tokens))
        if mask_slice is not None:
            mask_indices = mask_indices[mask_slice]
        for i in mask_indices:
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)  
        masks = hyper_in @ upscaled_embedding.transpose(-1, -2)

        iou_pred = self.iou_prediction_head(iou_token_out)
        if mask_slice is not None:
            iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred


# Adapted from https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
# Used in MaskDecoder for SAM
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x), inplace=True) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

