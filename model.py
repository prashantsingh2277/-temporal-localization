import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalTemporalBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(LocalTemporalBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        return F.relu(self.conv(x))

class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels):
        super(GlobalContextBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=2)

    def forward(self, x):
        
        x_perm = x.permute(2, 0, 1)
        attn_out, _ = self.attn(x_perm, x_perm, x_perm)
        return attn_out.permute(1, 2, 0)

class TemporalLocalizationModel(nn.Module):

    def __init__(self, in_channels=2048):
        super(TemporalLocalizationModel, self).__init__()
        half = in_channels // 2
        self.local_block = LocalTemporalBlock(half)
        self.global_block = GlobalContextBlock(half)
        self.fuse = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.boundary_head = nn.Conv1d(in_channels, 2, kernel_size=1)
        self.segment_head = nn.Conv1d(in_channels, 2, kernel_size=1)

    def forward(self, x):
        c = x.size(1)
        x_local = x[:, :c // 2, :]
        x_global = x[:, c // 2:, :]

        local_feat = self.local_block(x_local)      
        global_feat = self.global_block(x_global)   

        feat = torch.cat([local_feat, global_feat], dim=1)  
        feat = F.relu(self.fuse(feat))

        boundary_pred = self.boundary_head(feat)  
        segment_pred = self.segment_head(feat)    
        return boundary_pred, segment_pred
