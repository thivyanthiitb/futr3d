import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_

from .modified_cnw import ModifiedCNW
from .utils.pos_embed import SinePositionEmbedding
from .utils.ms_deformable_attn import MSDeformAttn

__all__ = ["AttentiveCNW"]

class BEVEvolvingBlock(nn.Sequential):
    def __init__(
        self,
        hidden_dim,
        is_cross_attn=True,
        ffn_dim=256,
        n_heads=8,
        n_points=8,
        dropout=0.1,
    ):
        super().__init__()
        self.is_cross_attn = is_cross_attn

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.attn = MSDeformAttn(hidden_dim, 1, n_heads, n_points)  # 1 is the number of levels

        self.norm2 = nn.LayerNorm(hidden_dim)

        # TODO: choose a proper FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.ReLU(),
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    @staticmethod
    def get_reference_points(spatial_shape, device):
        H, W = spatial_shape
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
        )
        ref = torch.stack((ref_x.reshape(-1), ref_y.reshape(-1)), -1)
        reference_points = ref[None, :, :]
        return reference_points
    

    def forward(self, tgt, query_pos=None, pos=None, src=None, reference_points=None, src_spatial_shape=None):  
        self.norm1.to(device=tgt.device)
        self.norm2.to(device=tgt.device)
        self.ffn.to(device=tgt.device)

        tgt2 = self.norm1(tgt)

        if self.is_cross_attn:
            # reference_points = self.reference_points(src_spatial_shape, src.device)
            reference_points = reference_points.unsqueeze(2)
            tgt2 = self.attn(
                tgt2 + query_pos, 
                reference_points.to(dtype=torch.float32), 
                src.to(dtype=torch.float32), 
                torch.tensor([src_spatial_shape], device=src.device), 
                torch.tensor([0], device=src.device)
            )
        else:
            # in deformable detr, src is used instead of pos but to keep conisitent 
            # with self.norm1(tgt), tgt is gonna provide source for this
            reference_points = self.get_reference_points(src_spatial_shape, tgt.device).unsqueeze(2)
            tgt2 = self.attn(
                tgt + pos, 
                reference_points.to(dtype=torch.float32), 
                tgt.to(dtype=torch.float32), 
                torch.tensor([src_spatial_shape], device=tgt.device), 
                torch.tensor([0], device=tgt.device)
            )
            
        tgt = tgt + tgt2

        tgt2 = self.norm2(tgt)
        tgt2 = self.ffn(tgt2)

        return tgt + tgt2


class AttentiveCNW(nn.Module):
    def __init__(
        self,
        num_cross_attn,
        num_self_attn,
        num_queries,
        out_channels=256,
        d_model=256,
        nhead=8,
        dropout=0.1,
        fusion_input_shape=None,
        use_adaptive_fusion=True,
    ) -> None:
        super().__init__()
        
        self.feature_shape = fusion_input_shape
        self.out_channels = out_channels
        self.reference_points = nn.Linear(d_model, 2)
        self.query_embed = nn.Embedding(num_queries, 2 * d_model)
        self.sinEmbed = SinePositionEmbedding(d_model // 2, normalize=True)

        self.fuser = ModifiedCNW(d_model, fusion_input_shape, use_adaptive_fusion)

        self.cross_attn_list = []
        self.self_attn_list = []

        for _ in range(num_cross_attn):
          self.cross_attn_list.append(
              BEVEvolvingBlock(
                  self.out_channels, 
                  is_cross_attn=True, 
                  n_heads=nhead, 
                  dropout=dropout, 
                )
            )

        for _ in range(num_self_attn):
          self.self_attn_list.append(
              BEVEvolvingBlock(
                  self.out_channels, 
                  is_cross_attn=False, 
                  n_heads=nhead, 
                  dropout=dropout, 
                )
            )

    def _reset_parameters(self):
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)

    def forward(self, features) -> torch.Tensor:
        bs = features[0].shape[0]
        h, w = self.feature_shape
        src_spatial_shape = [h, w]

        fused_features = self.fuser(features)

        # deformable detr doesnt seem to be using pos_embed
        pos_embed = self.sinEmbed(torch.zeros(bs, self.out_channels, h, w))
        pos_embed = pos_embed.to(fused_features.device)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)

        query_embed = self.query_embed.weight
        query_embed, tgt = torch.split(query_embed, self.out_channels, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

        reference_points = self.reference_points(query_embed).sigmoid()

        for cross_attn in self.cross_attn_list:
            tgt = cross_attn(
                tgt,
                src=fused_features,
                query_pos=query_embed,
                reference_points=reference_points,
                src_spatial_shape=src_spatial_shape,
            )

        for self_attn in self.self_attn_list:
            tgt = self_attn(tgt, pos=pos_embed, src_spatial_shape=src_spatial_shape)

        return tgt
