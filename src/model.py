import torch
import torch.nn as nn
import torch.nn.functional as F

class ResMLP(nn.Module):
    def __init__(self, dim, hidden_mult=4, dropout=0.15):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, int(dim * hidden_mult))
        self.fc2 = nn.Linear(int(dim * hidden_mult), dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        return x + self.drop(self.fc2(self.drop(F.gelu(self.fc1(self.ln(x))))))

class GenericMLPEncoder(nn.Module):
    def __init__(self, in_dim, width, out_dim, blocks=2, dropout=0.10):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, width), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResMLP(width, dropout=dropout) for _ in range(blocks)])
        self.out = nn.Sequential(nn.Linear(width, out_dim), nn.ReLU(inplace=True))
        
    def forward(self, x): 
        return self.out(self.blocks(self.proj(x)))

class ResBlock1D(nn.Module):
    def __init__(self, c, dropout=0.15):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(c)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)

class XRDEncoder(nn.Module):
    def __init__(self, in_ch=1, base=64, blocks=6, dropout=0.10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=9, padding=4), 
            nn.BatchNorm1d(base), 
            nn.ReLU(inplace=True)
        )
        layers = []
        for i in range(blocks):
            layers.append(ResBlock1D(base, dropout=dropout))
            if i % 2 == 1: layers.append(nn.MaxPool1d(2))
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x): 
        return self.pool(self.backbone(self.stem(x))).squeeze(-1)

class UnifiedMaterialsRegressor(nn.Module):
    def __init__(self, e_g1, e_g2, active_branches=None, dropout=0.10):
        super().__init__()
        self.active = active_branches or {"xrd": True, "tabular": True}
        
        self.xrd  = XRDEncoder(base=64, blocks=6, dropout=dropout)
        self.xrdf = GenericMLPEncoder(8, 128, 32, dropout=dropout)
        self.mag1 = GenericMLPEncoder(e_g1, 256, 64, dropout=dropout)
        self.mag2 = GenericMLPEncoder(e_g2, 256, 64, dropout=dropout)
        self.symm = GenericMLPEncoder(3, 64, 16, dropout=dropout)
        self.vol  = GenericMLPEncoder(1, 64, 16, dropout=dropout)
        self.geo  = GenericMLPEncoder(10, 64, 16, dropout=dropout)
        
        fused_dim = 64 + 32 + 64 + 64 + 16 + 16 + 16 
        
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            *[ResMLP(256, dropout=dropout) for _ in range(3)],
            nn.Linear(256, 1)
        )

    def forward(self, x_xrd, x_xrdf, x_g1, x_g2, x_symm, x_vol, x_geo):
        if not self.active["xrd"]:
            x_xrd, x_xrdf = torch.zeros_like(x_xrd), torch.zeros_like(x_xrdf)
        if not self.active["tabular"]:
            x_g1, x_g2 = torch.zeros_like(x_g1), torch.zeros_like(x_g2)
            x_symm, x_vol, x_geo = torch.zeros_like(x_symm), torch.zeros_like(x_vol), torch.zeros_like(x_geo)
        if x_xrd is not None and x_xrd.dim() == 2:
            x_xrd = x_xrd.unsqueeze(1)
        
        concat = torch.cat([
            self.xrd(x_xrd), self.xrdf(x_xrdf), 
            self.mag1(x_g1), self.mag2(x_g2), 
            self.symm(x_symm), self.vol(x_vol),
            self.geo(x_geo) 
        ], dim=1)
        return self.head(concat)