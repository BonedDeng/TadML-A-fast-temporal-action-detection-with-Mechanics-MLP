import torch
from torch import nn, Tensor
from torch.nn import functional as F
from timm.models.layers import DropPath

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv1d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv1d(hidden_dim, out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))
class PATM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_h = nn.Conv1d(dim, dim, 1)
        self.fc_w = nn.Conv1d(dim, dim, 1)
        self.fc_c = nn.Conv1d(dim, dim, 1)

        self.tfc_h = nn.Conv1d(2 * dim, dim, (1, 7), 1, (0, 7 // 2), groups=dim, bias=False)
        self.tfc_w = nn.Conv1d(2 * dim, dim, (7, 1), 1, (7 // 2, 0), groups=dim, bias=False)
        self.reweight = MLP(dim, dim // 4, dim * 3)

        self.proj = nn.Conv1d(dim, dim, 1)

        self.theta_h_conv = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            #nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.theta_w_conv = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            #nn.BatchNorm2d(dim),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H = x.shape

        theta_h = self.theta_h_conv(x)
        #theta_w = self.theta_w_conv(x)

        x_h = self.fc_h(x)
        #x_w = self.fc_w(x)
        c = self.fc_c(x)

        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        #x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)

        h = self.tfc_h(x_h)
        #w = self.tfc_w(x_w)

        a = F.adaptive_avg_pool2d(h  + c, output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        #x = h * a[0] + w * a[1] + c * a[2]
        x = h * a[0] + c * a[2]

        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4, dpr=0.):
        super().__init__()
        #self.norm1 = nn.BatchNorm2d(dim)
        self.attn = PATM(dim)


    def forward(self, x: Tensor) -> Tensor:
        x = self.attn(x)
        return x
