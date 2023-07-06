import torch
from torch import nn

from .TadML import register_backbone
from .blocks import (get_sinusoid_encoding,  MaskedConv1D,LayerNorm,MLPMixer, MaskedMLPMixer,ConvMixer,WaveMLP)



@register_backbone("mlp")
class MlpBackbone(nn.Module):
    """
        A backbone that with only conv
    """
    def __init__(
        self,
        n_in,               # input feature dimension
        n_embd,             # embedding dimension (after convolution)
        n_embd_ks,          # conv kernel size of the embedding network
        arch = (2, 2, 5),   # (#convs, #stem convs, #branch convs)
        scale_factor = 2,   # dowsampling rate for the branch
        with_ln=False,      # if to use layernorm
    ):
        super().__init__()
        assert len(arch) == 3
        self.arch = arch
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.embd_norm =LayerNorm(n_embd)

        self.WaveMLP = WaveMLP(model_name ='T',n_in=512,scale_factor=2)

        self.maskmlp = MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln) )
    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # print('out-----------',x.shape)
        # print('out-----------',mask.shape)
        x, mask = self.maskmlp(x,mask)
        x = self.relu(self.embd_norm(x))
        #2d wave-mlp
        out_feats = tuple()
        out_masks = tuple()
        out_feats += (x, )
        out_masks += (mask, )
        # 1x resolution
        for i in range(self.arch[2]):
            x = torch.stack((x, x), dim=3)
            mask = torch.stack((mask, mask), dim=3)
            x, mask = self.WaveMLP(x, mask)
            x = torch.squeeze(x, dim=3)
            mask = torch.squeeze(mask, dim=3)

            out_feats += (x, )
            out_masks += (mask, )
            # if i < (self.arch[2]-1):
            #     x = torch.squeeze(x,dim=3)
            #     mask = torch.squeeze(mask,dim=3)
        return out_feats, out_masks

