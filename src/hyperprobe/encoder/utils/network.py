import torch
import torch.nn as nn

class NetworkModel(nn.Module):
    def __init__(self, input_dim, output_dim, drop_p = 0):
        super(NetworkModel, self).__init__()
        
        # External parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_p = drop_p
        
        # Local parameters
        self.latent_dim = 4096
        num_blocks = 2
        num_heads = 0
        
        # Input Processing: Map input to latent space.
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
        )
        
        # Residual Blocks: Stack a few residual blocks.
        self.blocks = nn.ModuleList([
            ResidualBlock(self.latent_dim, drop_p = drop_p, num_heads = num_heads if isinstance(num_heads, int) else num_heads[i]) 
            for i in range(num_blocks)
        ])
        
        # Output Projection: Map latent space to output.
        self.output_layer = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, self.output_dim),
            nn.Tanh()
        )
        
        # Define Model Name for Identification
        configs = [f'equal{len(self.blocks)}']
        if isinstance(num_heads, int) and num_heads > 0:
            configs.append(f'att{num_heads}')
        elif isinstance(num_heads, list):
            configs.append('att'+'&'.join([str(num) for num in num_heads]))
        self.name = '_'.join(configs)
        
    def forward(self, x):
        
        # Input layer
        x = self.input_layer(x)
        
        # Process each residual block.
        for block in self.blocks:
            x = block(x)
            
        # Output layer
        out = self.output_layer(x)
        
        return out 

class ResidualBlock(nn.Module):
    def __init__(self, dim, drop_p=0, num_heads=0):
        super().__init__()
        
        # Main transformation path.
        self.main_path = nn.Sequential(
            nn.Linear(dim, dim),    
            nn.GELU(),
            nn.LayerNorm(dim)
        )
        # Optional Self-Attention: Only if more than 0 heads.
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop_p, batch_first=True) if num_heads > 0 else None
        
        # If attention is defined, add dropout and a separate learnable scale factor.
        if self.attn is not None:
            self.attn_dropout = nn.Dropout(drop_p)
        
        # Projection (identity if dimensions match).
        self.residual_proj = nn.Identity()
        self.dropout = nn.Dropout(drop_p)
        
        # Learnable scale for the residual.
        self.scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        
        # Store the residual.
        identity = x
        
        # Main transformation path.
        out = self.main_path(x)
        
        # Apply self-attention if defined.
        if self.attn is not None:
            
            # Add sequence dimension for attention.
            out_unsq = out.unsqueeze(1)
            attn_out, _ = self.attn(out_unsq, out_unsq, out_unsq)
            
            # Apply attention dropout and scale the output.
            attn_out = self.attn_dropout(attn_out)
            out = attn_out.squeeze(1)
            
        # Combine the transformed representation with the residual.
        out = out + self.residual_proj(identity) * self.scale
        
        return self.dropout(out)