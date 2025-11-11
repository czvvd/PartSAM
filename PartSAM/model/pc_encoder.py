# https://github.com/baaivision/Uni3D/blob/main/models/point_encoder.py
from typing import Union

import timm
import torch
import torch.nn as nn
from timm.models.eva import Eva
from timm.models.vision_transformer import VisionTransformer

from .common import KNNGrouper, NNGrouper, PatchEncoder

from partfield.model.UNet.model import ResidualUNet3D
from partfield.model.triplane import TriplaneTransformer, get_grid_coord 
from partfield.model.model_utils import VanillaMLP
from partfield.model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat
import numpy as np

def smart_load_state_dict(model, pretrained_dict):
    model_dict = model.state_dict()
    
    # 1. 过滤掉维度不匹配的权重
    matched_dict = {
        k: v for k, v in pretrained_dict.items() 
        if k in model_dict and v.shape == model_dict[k].shape
    }
    
    # 2. 记录不匹配的键（调试用）
    missing_keys = [k for k in pretrained_dict if k not in model_dict]
    shape_mismatch_keys = [
        k for k in pretrained_dict 
        if k in model_dict and pretrained_dict[k].shape != model_dict[k].shape
    ]
    
    # 3. 加载匹配的权重
    model.load_state_dict(matched_dict, strict=False)
    
    # 打印调试信息
    print(f"Successfully loaded {len(matched_dict)}/{len(pretrained_dict)} parameters")
    if missing_keys:
        print(f"Missing keys (ignored): {missing_keys}")
    if shape_mismatch_keys:
        print(f"Shape mismatch keys (ignored): {shape_mismatch_keys}")
    
    return model
    
class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_patches,
        patch_size,
        radius: float = None,
        centralize_features=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.grouper = KNNGrouper(
            num_patches,
            patch_size,
            radius=radius,
            centralize_features=centralize_features,
        )

        self.patch_encoder = PatchEncoder(in_channels, out_channels, [448, 448])
        # self.patch_encoder = PatchEncoder(in_channels, out_channels, [128, 512])

    def forward(self, coords: torch.Tensor, features: torch.Tensor):
        patches = self.grouper(coords, features)
        patch_features = patches["features"]  # [B, L, K, C_in]
        x = self.patch_encoder(patch_features)
        patches["embeddings"] = x
        return patches

class PatchDropout(nn.Module):
    """Randomly drop patches.

    References:
    - https://arxiv.org/abs/2212.00794
    - `timm.layers.patch_dropout`. It uses `argsort` rather than `topk`, which might be inefficient.
    """

    def __init__(self, prob, num_prefix_tokens: int = 1):
        super().__init__()
        assert 0.0 <= prob < 1.0, prob
        self.prob = prob
        # exclude CLS token (or other prefix tokens)
        self.num_prefix_tokens = num_prefix_tokens

    def forward(self, x: torch.Tensor):
        # x: [B, L, ...]
        if not self.training or self.prob == 0.0:
            return x

        if self.num_prefix_tokens:
            prefix_tokens = x[:, : self.num_prefix_tokens]
            x = x[:, self.num_prefix_tokens :]
        else:
            prefix_tokens = None

        B, L = x.shape[:2]
        num_keep = max(1, int(L * (1.0 - self.prob)))
        rand = torch.randn(B, L, device=x.device)
        keep_indices = rand.topk(num_keep, dim=1).indices
        _keep_indices = keep_indices.reshape((B, num_keep) + (-1,) * (x.dim() - 2))
        _keep_indices = _keep_indices.expand((-1, -1) + x.shape[2:])
        x = x.gather(1, _keep_indices)

        if prefix_tokens is not None:
            x = torch.cat((prefix_tokens, x), dim=1)

        return x


class PFEncoderDual(nn.Module):
    def __init__(
        self,
        patch_embed: PatchEmbed
    ):
        super().__init__()
        # Patch embedding
        self.patch_embed = patch_embed

        # Transformer encoder
        self.partfield = PartField()
        self.partfieldMy = PartFieldPath()


    def forward(self, coords, color, normal,only_pf=False):
        # Get triplane features
        planes1 = self.partfield(coords)
        planes2 = self.partfieldMy(coords, color, normal)
        if only_pf:
            planes = planes1
        else:
            planes = (planes1 + planes2) /2.0
        sdf_planes, part_planes = torch.split(planes, [64, planes.shape[2] - 64], dim=2)

        pf_feat = sample_triplane_feat(part_planes, coords)

        patches = self.patch_embed(coords, pf_feat)
        if isinstance(patches, list):
            centers = patches[-1]["centers"]
        else:
            centers = patches["centers"]  # [B, L, 3]


        return patches, pf_feat, part_planes
    


class PartField(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

        # Transformer encoder
        self.pvcnn = TriPlanePC2Encoder(
                point_encoder_type='pvcnn',
                z_triplane_channels=256,
                z_triplane_resolution=128,
                device="cuda",
                shape_min=-1, 
                shape_length=2,
                use_2d_feat=False,
                is_My=False)
        self.triplane_transformer = TriplaneTransformer(
            input_dim=128 * 2,
            transformer_dim=1024,
            transformer_layers=6,
            transformer_heads=8,
            triplane_low_res=32,
            triplane_high_res=128,
            triplane_dim=512,
        )


    def forward(self, coords):

        # Get triplane features
        pc_feat = self.pvcnn(coords,coords)
        # pc_feat = self.pvcnn(coords,coords, add_features=normal)
        planes = self.triplane_transformer(pc_feat)

        return planes
    
class PartFieldPath(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

        # Transformer encoder
        self.pvcnn = TriPlanePC2Encoder(
                point_encoder_type='pvcnn',
                z_triplane_channels=256,
                z_triplane_resolution=128,
                device="cuda",
                shape_min=-1, 
                shape_length=2,
                use_2d_feat=False,
                is_My=True)
        self.triplane_transformer = TriplaneTransformer(
            input_dim=128 * 2,
            transformer_dim=1024,
            transformer_layers=6,
            transformer_heads=8,
            triplane_low_res=32,
            triplane_high_res=128,
            triplane_dim=512,
        )


    def forward(self, coords, color, normal):

        # Get triplane features
        pc_feat = self.pvcnn(coords,coords, add_features=torch.cat([color,normal],dim=2))
        # pc_feat = self.pvcnn(coords,coords, add_features=normal)
        planes = self.triplane_transformer(pc_feat)

        return planes



class Block(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels):
        super().__init__()
        # Follow timm.layers.mlp
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_channels),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        # PreLN. Follow timm.models.vision_transformer
        return x + self.mlp(self.norm(x))


class PatchEmbedNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, num_patches) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        hidden_dim = hidden_dim or out_channels

        self.grouper = NNGrouper(num_patches)
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.blocks1 = nn.Sequential(
            *[Block(hidden_dim, hidden_dim, hidden_dim) for _ in range(3)]
        )
        self.blocks2 = nn.Sequential(
            *[Block(hidden_dim, hidden_dim, hidden_dim) for _ in range(3)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_channels)

    def forward(self, coords: torch.tensor, features: torch.tensor):
        patches = self.grouper(coords, features)
        patch_features = patches["features"]  # [B, N, D]
        nn_idx = patches["nn_idx"]  # [B, N]

        x = self.in_proj(patch_features)
        x = self.blocks1(x)  # [B, N, D]
        y = x.new_zeros(x.shape[0], self.grouper.num_groups, x.shape[-1])
        y.scatter_reduce_(
            1, nn_idx.unsqueeze(-1).expand_as(x), x, "amax", include_self=False
        )
        x = self.blocks2(y)
        x = self.norm(x)
        x = self.out_proj(x)
        patches["embeddings"] = x
        return patches


class PatchEmbedHier(nn.Module):
    """PointNet++ style with hierarchical grouping."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_patches: list[int],
        patch_size: list[int],
        radius: list[float] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.grouper1 = KNNGrouper(
            num_patches[0],
            patch_size[0],
            radius=radius[0] if radius else None,
        )
        self.patch_encoder1 = PatchEncoder(in_channels, 128, [64, 128])

        self.grouper2 = KNNGrouper(
            num_patches[1],
            patch_size[1],
            radius=radius[1] if radius else None,
        )
        self.patch_encoder2 = PatchEncoder(128 + 3, out_channels, [128, 256])

    def forward(self, coords: torch.Tensor, features: torch.Tensor):
        patches1 = self.grouper1(coords, features)
        x1 = self.patch_encoder1(patches1["features"])
        patches1["embeddings"] = x1

        patches2 = self.grouper2(patches1["centers"], x1, use_fps=False)
        x2 = self.patch_encoder2(patches2["features"])
        patches2["embeddings"] = x2

        return [patches1, patches2]

