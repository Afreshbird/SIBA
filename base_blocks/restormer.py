import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers


# Restormer block
class TransformerBlock_SA(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 num_heads=8,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False,):
        super(TransformerBlock_SA, self).__init__()

        if out_channel != in_channel:
            self.conv = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        else:
            self.conv = nn.Identity()

        self.norm1 = LayerNorm(out_channel, 'WithBias')
        self.attn = Self_Attention(out_channel, num_heads=num_heads, qkv_bias=qkv_bias,)

        self.norm2 = LayerNorm(out_channel, 'WithBias')
        self.mlp = Mlp(in_features=out_channel,ffn_expansion_factor=ffn_expansion_factor,)

        
    def forward(self, x): 
        x = self.conv(x)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x)) 
        return x



# I/V-SCA
class TransformerBlock_CA(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 num_heads=8,
                 ffn_expansion_factor=1.,  
                 qkv_bias=False):
        super(TransformerBlock_CA, self).__init__()

        if out_channel != in_channel:
            self.conv = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        else:
            self.conv = nn.Identity()

        self.norm1 = LayerNorm(out_channel, 'WithBias')
        self.attn = Cross_Attention(out_channel, num_heads=num_heads, qkv_bias=qkv_bias,)

        self.norm2 = LayerNorm(out_channel, 'WithBias')
        self.mlp = Mlp(in_features=out_channel,ffn_expansion_factor=ffn_expansion_factor,)

        
    def forward(self, x, weight): 
        x = self.conv(x)
        x = x + self.attn(self.norm1(x), weight)
        x = x + self.mlp(self.norm2(x)) 
        return x



class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



# Multi-DConv Head Transposed Self-Attention (MDTA)
class Self_Attention(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(Self_Attention, self).__init__()

        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1)) # torch.Size([8, 1, 1])
        self.qkv1 = nn.Conv2d(dim, dim*3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):

        b, c, h, w = x.shape

        qkv = self.qkv2(self.qkv1(x))

        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)

        return out



# Dual-Dconv Infrared/Visible Cross Attention (I/V-DCA)
class Cross_Attention(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=False,):
        super(Cross_Attention, self).__init__()

        self.num_heads = num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim*2, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, kv, q):

        b, c, h, w = kv.shape

        kv = self.qkv2(self.qkv1(kv))

        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
     

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)

        return out



# Gated-Dconv Feed-Forward Network (GDFN)
class Mlp(nn.Module):
    def __init__(self, 
                 in_features, 
                 ffn_expansion_factor = 2,
                 bias = False):
        super().__init__()

        hidden_features = int(in_features*ffn_expansion_factor)
        self.project_in = nn.Conv2d(in_features, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=bias)
    
    def forward(self, x): 

        x = self.project_in(x) 
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x) 

        return x



class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


