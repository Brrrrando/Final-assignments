import torch 
import torch.nn as nn
from layers import *

class VisionTransformer(nn.Module):
    def __init__(
            self, 
            image_size = 384,
            patch_size = 16, 
            in_channels  =3, 
            n_classes = 1000, 
            embedding_dimension=768,
            depth=12,
            n_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            proj_p=0.,
            attn_p=0
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            image_size = image_size,
            patch_size=patch_size, 
            in_channels=in_channels,
            embedding_dimension=embedding_dimension
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dimension))
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embedding.n_patches, embedding_dimension)
        )
        self.pos_dropout = nn.Dropout(p=proj_p)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
            dimensions=embedding_dimension,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias = qkv_bias,
            proj_p=proj_p,
            attn_p=attn_p
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embedding_dimension, eps=1e-6)
        self.head = nn.Linear(embedding_dimension, n_classes)
    
    def forward(self, x ):
        n_samples = x.shape[0]
        x = self.patch_embedding(x)
        cls_token = self.cls_token.expand(
            n_samples, -1, -1
        )
        x = torch.cat((cls_token, x), dim=1) 
        x = x + self.pos_embedding
        x = self.pos_dropout(x)
        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_embedding = x[:, 0]
        output = self.head(cls_token_embedding)
        return output
    
