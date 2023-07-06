import math
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn,Tensor
from .weight_init import trunc_normal_

from einops.layers.torch import Rearrange

########mlp
class PreNormResidualc(nn.Module):
    def __init__(self, dim, fn):
        super(PreNormResidualc, self).__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class PreNormResiduall(nn.Module):
    def __init__(self, dim, fn):
        super(PreNormResiduall, self).__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = Rearrange('b l c-> b c l')(x)
        x = self.norm(x)
        x = Rearrange('b c l -> b l c')(x)
        return self.fn(x) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.Softplus(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


def FeedForwardinout(in_channel, out_channel, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(in_channel, out_channel * expansion_factor),
        nn.Softplus(),
        nn.Dropout(dropout),
        dense(out_channel * expansion_factor, out_channel),
        nn.Dropout(dropout)
    )


def MLPMixer(seq_len, in_channel, out_channel, depth, expansion_factor=4, dropout=0., scale_factor=2):
    assert seq_len % scale_factor == 0, 'seq_len % scale_factor is not None!'
    scale_dim = seq_len // scale_factor
    return nn.Sequential(
        Rearrange('b c l -> b l c'),
        FeedForwardinout(in_channel, out_channel, expansion_factor, dropout, nn.Linear),
        *[nn.Sequential(
            PreNormResiduall(out_channel, FeedForward(out_channel, expansion_factor, dropout, nn.Linear)),
            Rearrange('b l c -> b c l'),
            PreNormResidualc(out_channel, FeedForward(seq_len, expansion_factor, dropout, nn.Linear)),
            Rearrange('b c l -> b l c')
        ) for _ in range(depth)],
        Rearrange('b l c -> b c l'),
        nn.Linear(seq_len, scale_dim),
        LayerNorm(out_channel)
    )
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Conv2d(2048, dim, kernel_size=kernel_size, stride=patch_size, padding=3),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, 1, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)],
        )

        #self.classifier = nn.Sequential(
        #   nn.AdaptiveAvgPool2d((1, 1))#)


    def forward(self, x):
        embedding = self.embedding(x)
        print(embedding.shape)
        embedding = self.blocks(embedding)
        #out = self.classifier(embedding)
        out = embedding
        return out

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))

class PATM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_h = nn.Conv2d(dim, dim, 1)
        self.fc_w = nn.Conv2d(dim, dim, 1)
        self.fc_c = nn.Conv2d(dim, dim, 1)

        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), 1, (0, 7 // 2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), 1, (7 // 2, 0), groups=dim, bias=False)
        self.reweight = MLP(dim, dim // 4, dim * 3)

        self.proj = nn.Conv2d(dim, dim, 1)

        self.theta_h_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.theta_w_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape

        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)

        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        c = self.fc_c(x)

        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)

        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)

        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4, dpr=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = PATM(dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbedOverlap(nn.Module):
    """Image to Patch Embedding with overlapping
    """

    def __init__(self, n_in = 2048,patch_size=16, stride=16, padding=0, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(n_in, embed_dim, patch_size, stride, padding)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> Tensor:
        return self.norm(self.proj(x))

class Downsample(nn.Module):
    """Downsample transition stage"""

    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 3, 1, 1)
        self.norm = nn.BatchNorm2d(c2)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))

class Downsample_c(nn.Module):
    """Downsample transition stage"""

    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 3, 2, 1)
        self.norm = nn.BatchNorm2d(c2)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x))

wavemlp_settings = {
    'T': [[1, 1, 1, 1], [4, 4, 4, 4]],  # [layers]
    'S': [[2, 3, 10, 3], [4, 4, 4, 4]],
    'M': [[3, 4, 18, 3], [8, 8, 4, 4]]
}
class WaveMLP(nn.Module):
    def __init__(self, n_in,scale_factor,model_name: str = 'T', pretrained: str = None) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        assert model_name in wavemlp_settings.keys(), f"WaveMLP model name should be in {list(wavemlp_settings.keys())}"
        layers, mlp_ratios = wavemlp_settings[model_name]
        embed_dims = [64, 128, 320, n_in]

        self.patch_embed = PatchEmbedOverlap(n_in=n_in,patch_size = 1, stride=1, padding =0, embed_dim = embed_dims[0])

        network = []

        for i in range(len(layers)):
            stage = nn.Sequential(*[
                Block(embed_dims[i], mlp_ratios[i])
                for _ in range(layers[i])])

            network.append(stage)
            if i >= len(layers) - 1: break
            network.append(Downsample(embed_dims[i], embed_dims[i + 1]))

        self.network = nn.ModuleList(network)
        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.downsample = Downsample_c(embed_dims[-1], embed_dims[-1])
        # use as a backbone
        self.out_indices = [0, 2, 4, 6]
        # for i, layer in enumerate(self.out_indices):
        #     self.add_module(f"norm{layer}", nn.BatchNorm2d(embed_dims[i]))

        self._init_weights(pretrained)

    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu')['model'])
        else:
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    if n.startswith('head'):
                        nn.init.zeros_(m.weight)
                        nn.init.zeros_(m.bias)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def return_features(self, x):
        x = self.patch_embed(x)
        outs = []

        for i, blk in enumerate(self.network):
            x = blk(x)
            if i in self.out_indices:
                out = getattr(self, f"norm{i}")(x)
                outs.append(out)
        return outs

    def forward(self, x: torch.Tensor,mask):
        x = self.patch_embed(x)

        for blk in self.network:
            x = blk(x)
        out_mlp = self.norm(x)
        if self.scale_factor > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.float(),
                size=(mask.size()[2] // self.scale_factor,mask.size()[3] // self.scale_factor),
                mode='bilinear'
            )
            out_mlp = self.downsample(out_mlp)
        else:
            out_mask = mask.float()
        #print('om',out_mlp.shape,out_mask.shape)
        out_mlp = out_mlp * out_mask.detach()
        out_mask = out_mask.bool()
        return out_mlp, out_mask

class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.float(),
                size=T // self.stride,
                mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.float()

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


class MaskedMHA(nn.Module):
    """
    Multi Head Attention with mask

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the input embedding
            n_head,  # number of heads in multi-head self-attention
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # calculate query, key, values for all heads in batch
        # (B, nh * hs, T)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # move head forward to be the batch dim
        # (B, nh * hs, T) -> (B, nh, T, hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = att @ (v * mask[:, :, :, None].float())
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * mask.float()
        return out, mask

class MaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the output features
            n_head,  # number of heads in multi-head self-attention
            n_qx_stride=1,  # dowsampling stride for query and input
            n_kv_stride=1,  # downsampling stride for key and value
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].float())
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.float()
        return out, qx_mask

class LocalMaskedMHCA(nn.Module):
    """
    Local Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    The implementation is fairly tricky, code reference from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
            self,
            n_embd,  # dimension of the output features
            n_head,  # number of heads in multi-head self-attention
            window_size,  # size of the local attention window
            n_qx_stride=1,  # dowsampling stride for query and input
            n_kv_stride=1,  # downsampling stride for key and value
            attn_pdrop=0.0,  # dropout rate for the attention map
            proj_pdrop=0.0,  # dropout rate for projection op
            use_rel_pe=False  # use relative position encoding
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap = window_size // 2
        # must use an odd window size
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # relative position encoding
        if self.use_rel_pe:
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / self.n_embd) ** 0.5)

    @staticmethod
    def _chunk(x, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # x: B x nh, T, hs
        # non-overlapping chunks of size = 2w -> B x nh, T//2w, 2w, hs
        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        # B x nh, #chunks = T//w - 1, 2w, hs
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        """pads rows and then flips rows and columns"""
        # padding value is not important because it will be overwritten
        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        # `== 1` converts to bool or uint8
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        # `== 1` converts to bool or uint8
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example::
              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
        x = nn.functional.pad(
            x, (0, window_overlap + 1)
        )
        # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        x = x.view(total_num_heads, num_chunks, -1)
        # total_num_heads x num_chunks x window_overlap*window_overlap
        x = x[:, :, :-window_overlap]
        x = x.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(
            self, query, key, num_heads, window_overlap
    ):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This implementation splits the input into overlapping chunks of size 2w with an overlap of size w (window_overlap)
        """
        # query / key: B*nh, T, hs
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # B * num_heads, head_dim, #chunks=(T//w - 1), 2w
        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        # convert diagonals into columns
        # B * num_heads, #chunks, 2w, 2w+1
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                                :, :, :window_overlap, : window_overlap + 1
                                                                ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                               :, -1, window_overlap:, : window_overlap + 1
                                                               ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
                                                               :, :, -(window_overlap + 1): -1, window_overlap + 1:
                                                               ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
                                                                              :, 0, : window_overlap - 1,
                                                                              1 - window_overlap:
                                                                              ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
            self, attn_probs, value, num_heads, window_overlap
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # step 1: depth convolutions
        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # step 2: query, key, value transforms & reshape
        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        # (B, nh * hs, T) -> (B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # view as (B * nh, T, hs)
        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        # step 3: compute local self-attention with rel pe and masking
        q *= self.scale
        # chunked query key attention -> B, T, nh, 2w+1 = window_size
        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)

        # rel pe
        if self.use_rel_pe:
            att += self.rel_pe
        # kv_mask -> B, T'', 1
        inverse_kv_mask = torch.logical_not(
            kv_mask[:, :, :, None].view(B, -1, 1))
        # 0 for valid slot, -inf for masked ones
        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(
            inverse_kv_mask, -1e4)
        # compute the diagonal mask (for each local window)
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask,
            1,
            self.window_overlap
        )
        att += diagonal_mask

        # ignore input masking for now
        att = nn.functional.softmax(att, dim=-1)
        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        att = att.masked_fill(
            torch.logical_not(kv_mask.squeeze(1)[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        # step 4: compute attention value product + output projection
        # chunked attn value product -> B, nh, T, hs
        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)
        # transpose to B, nh, hs, T -> B, nh*hs, T
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.float()
        return out, qx_mask



class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)





class MaskedMLPMixer(nn.Module):
    def __init__(
            self,
            seq_len,  # max_lens
            in_channel,  #
            out_channel,
            depth,
            expansion_factor=4,
            dropout=0.,
            scale_factor=1,
            bias=True
    ):
        super().__init__()
        assert seq_len % scale_factor == 0, 'seq_len % scale_factor is not None!'
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.depth = depth
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.scale_factor = scale_factor
        self.scale_dim = seq_len // scale_factor
        self.mlp = nn.Sequential(
            Rearrange('b c l -> b l c'),
            FeedForwardinout(in_channel, out_channel, expansion_factor, dropout, nn.Linear),
            *[nn.Sequential(
                PreNormResiduall(out_channel, FeedForward(out_channel, expansion_factor, dropout, nn.Linear)),
                Rearrange('b l c -> b c l'),
                PreNormResidualc(out_channel, FeedForward(seq_len, expansion_factor, dropout, nn.Linear)),
                Rearrange('b c l -> b l c')
            ) for _ in range(depth)],
            Rearrange('b l c -> b c l'),
            nn.Linear(seq_len, self.scale_dim),
            nn.Softplus(),
            LayerNorm(out_channel)
        )
        if bias:
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.constant_(m.bias, 0.)

    def forward(self, x, mask):
        B, C, T = x.size()
        # mlp
        out_mlp = self.mlp(x)
        # compute mask
        if self.scale_factor > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.float(),
                size=T // self.scale_factor,
                mode='nearest'
            )
        else:
            out_mask = mask.float()
        out_mlp = out_mlp * out_mask.detach()
        out_mask = out_mask.bool()
        return out_mlp, out_mask


class MaskedMLPMixer_c(nn.Module):
    def __init__(
            self,
            in_channel,  #
            out_channel,
            depth,
            expansion_factor=4,
            dropout=0.,
            scale_factor=1,
            bias=True
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.depth = depth
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.scale_factor = scale_factor
        self.bias = bias

    def forward(self, x, mask):
        # init by x.size
        B, C, T = x.size()
        seq_len = T
        scale_factor = self.scale_factor
        self.scale_dim = seq_len // scale_factor
        assert seq_len % scale_factor == 0, 'seq_len % scale_factor is not None!'
        scale_dim = seq_len // scale_factor

        in_channel = self.in_channel
        out_channel = self.out_channel
        depth = self.depth
        expansion_factor = self.expansion_factor
        dropout = self.dropout
        bias = self.bias
        self.mlp = nn.Sequential(
            Rearrange('b c l -> b l c'),
            FeedForwardinout(in_channel, out_channel, expansion_factor, dropout, nn.Linear),
            *[nn.Sequential(
                PreNormResiduall(out_channel, FeedForward(out_channel, expansion_factor, dropout, nn.Linear)),
                Rearrange('b l c -> b c l'),
                PreNormResidualc(out_channel, FeedForward(seq_len, expansion_factor, dropout, nn.Linear)),
                Rearrange('b c l -> b l c')
            ) for _ in range(depth)],
            Rearrange('b l c -> b c l'),
            LayerNorm(out_channel)
        )
        if bias:
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.constant_(m.bias, 0.)

        out_mlp = self.mlp(x)
        # compute mask
        if self.scale_factor > 1:
            dense = nn.Linear(seq_len, scale_dim)
            out_mlp = dense(out_mlp)
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.float(),
                size=T // self.scale_factor,
                mode='nearest'
            )
        else:
            out_mask = mask.float()
        return out_mlp, out_mask


class MaskedMLPMixer_post(nn.Module):
    def __init__(
            self,
            seq_len,  # max_lens
            in_channel,  #
            out_channel,
            depth,
            expansion_factor=4,
            dropout=0.,
            scale_factor=1,
            bias=True
    ):
        super().__init__()
        assert seq_len % scale_factor == 0, 'seq_len % scale_factor is not None!'
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.depth = depth
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.scale_factor = scale_factor
        self.scale_dim = seq_len // scale_factor
        self.mlp = nn.Sequential(
            Rearrange('b c l -> b l c'),
            FeedForwardinout(in_channel, out_channel, expansion_factor, dropout, nn.Linear),
            *[nn.Sequential(
                PreNormResiduall(out_channel, FeedForward(out_channel, expansion_factor, dropout, nn.Linear)),
                Rearrange('b l c -> b c l'),
                PreNormResidualc(out_channel, FeedForward(seq_len, expansion_factor, dropout, nn.Linear)),
                Rearrange('b c l -> b l c')
            ) for _ in range(depth)],
            Rearrange('b l c -> b c l'),
            # nn.Linear(seq_len, self.scale_dim),
            LayerNorm(out_channel)
        )
        self.scale = nn.ModuleList()
        self.scale.append(Scale())
        bias_value = -(math.log((1 - 1e-6) / 1e-6))
        if bias:
            for m, n in self.mlp.named_parameters():
                if 'bias' in m:
                    torch.nn.init.constant_(n, bias_value)

    def forward(self, x, mask):
        B, C, T = x.size()
        # mlp
        out_mlp = self.mlp(x)
        # compute mask
        out_mlp = F.relu(out_mlp)
        if self.scale_factor > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.float(),
                size=T // self.scale_factor,
                mode='nearest'
            )
        else:
            out_mask = mask.float()
        return out_mlp, out_mask


class MaskedMLPMixer_cls(nn.Module):
    def __init__(
         self,
         seq_len,       # max_lens
         in_channel,    #
         out_channel,
         depth,
         expansion_factor = 4,
         dropout = 0.,
         scale_factor = 1,
         bias=True
    ):
        super().__init__()
        assert seq_len % scale_factor == 0, 'seq_len % scale_factor is not None!'
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.depth = depth
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.scale_factor = scale_factor
        self.scale_dim = seq_len // scale_factor
        self.mlp = nn.Sequential(
            Rearrange('b c l -> b l c'),
            *[nn.Sequential(
                PreNormResiduall(in_channel, FeedForward(in_channel, expansion_factor, dropout, nn.Linear)),
                Rearrange('b l c -> b c l'),
                PreNormResidualc(in_channel, FeedForward(seq_len, expansion_factor, dropout, nn.Linear)),
                Rearrange('b c l -> b l c')
            ) for _ in range(depth)],
            Rearrange('b l c -> b c l'),
            nn.Linear(seq_len, self.scale_dim),
            LayerNorm(out_channel)
        )
        self.cls = nn.Sequential(
            Rearrange('b c l -> b l c'),
            FeedForwardinout(in_channel, out_channel, expansion_factor, dropout, nn.Linear),
            *[nn.Sequential(
                PreNormResiduall(out_channel, FeedForward(out_channel, expansion_factor, dropout, nn.Linear)),
                Rearrange('b l c -> b c l'),
                PreNormResidualc(out_channel, FeedForward(seq_len, expansion_factor, dropout, nn.Linear)),
                Rearrange('b c l -> b l c')
            ) for _ in range(depth)],
            Rearrange('b l c -> b c l'),
            nn.Linear(self.scale_dim, self.scale_dim),
            LayerNorm(in_channel)
        )
        bias_value = -(math.log((1 - 1e-6) / 1e-6))
        if bias:
            for m,n in self.mlp.named_parameters():
                if 'bias' in m:
                    torch.nn.init.constant_(n, bias_value)
    def forward(self, x, mask):
        B, C, T = x.size()
        #mlp
        out_mlp = self.mlp(x)
        # compute mask
        out_mlp = F.relu(out_mlp)
        if self.scale_factor > 1:
            #downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.float(),
                size = T // self.scale_factor,
                mode = 'nearest'
            )
        else:
            out_mask = mask.float()
        out_mlp = out_mlp * out_mask.detach()
        out_mlp = self.cls(out_mlp)
        out_mlp = out_mlp * out_mask.detach()
        out_mask = out_mask.bool()
        return out_mlp, out_mask

class MaskedMLPMixer_reg(nn.Module):
    def __init__(
         self,
         seq_len,       # max_lens
         in_channel,    #
         out_channel,
         depth,
         expansion_factor = 4,
         dropout = 0.,
         scale_factor = 1,
         bias=True
    ):
        super().__init__()
        assert seq_len % scale_factor == 0, 'seq_len % scale_factor is not None!'
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.depth = depth
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        self.scale_factor = scale_factor
        self.scale_dim = seq_len // scale_factor
        self.mlp = nn.Sequential(
            Rearrange('b c l -> b l c'),
            *[nn.Sequential(
                PreNormResiduall(in_channel, FeedForward(in_channel, expansion_factor, dropout, nn.Linear)),
                Rearrange('b l c -> b c l'),
                PreNormResidualc(in_channel, FeedForward(seq_len, expansion_factor, dropout, nn.Linear)),
                Rearrange('b c l -> b l c')
            ) for _ in range(depth)],
            Rearrange('b l c -> b c l'),
            nn.Linear(seq_len, self.scale_dim),
            LayerNorm(in_channel)
        )
        init_value = 1.0
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )
        self.reg = nn.Sequential(
            Rearrange('b c l -> b l c'),
            FeedForwardinout(in_channel, out_channel, expansion_factor, dropout, nn.Linear),
            *[nn.Sequential(
                PreNormResiduall(out_channel, FeedForward(out_channel, expansion_factor, dropout, nn.Linear)),
                Rearrange('b l c -> b c l'),
                PreNormResidualc(out_channel, FeedForward(seq_len, expansion_factor, dropout, nn.Linear)),
                Rearrange('b c l -> b l c')
            ) for _ in range(depth)],
            Rearrange('b l c -> b c l'),
            nn.Linear(self.scale_dim, self.scale_dim),
            LayerNorm(out_channel)
        )
        bias_value = -(math.log((1 - 1e-6) / 1e-6))
        if bias:
            for m,n in self.mlp.named_parameters():
                if 'bias' in m:
                    torch.nn.init.constant_(n, bias_value)
    def forward(self, x, mask):
        B, C, T = x.size()
        #mlp
        out_mlp = self.mlp(x)
        # compute mask
        out_mlp = F.relu(out_mlp)
        out_mlp = F.relu(self.scale *self.reg(out_mlp))
        if self.scale_factor > 1:
            #downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.float(),
                size = T // self.scale_factor,
                mode = 'nearest'
            )
        else:
            out_mask = mask.float()
        out_mlp = out_mlp * out_mask.detach()
        out_mlp = self.reg(out_mlp)
        out_mlp = out_mlp * out_mask.detach()
        out_mask = out_mask.bool()
        return out_mlp, out_mask

