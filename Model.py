import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.utils.data
import torch.nn.functional as f
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# Cluster ====================================================================

data = np.load('Original3Clear.npy')
# downscaled = np.load('Downscaled.npy')

tensor_y = torch.Tensor(data)

# tensor_x= torch.Tensor(downscaled)

y = tensor_y.view(tensor_y.size(0),1,
                      tensor_y.size(1),
                      tensor_y.size(2))

x = f.interpolate(y, (int(y.shape[2]/6) , 
                                        int(y.shape[3]/6)), 
                          mode='bicubic')

# x = tensor_x.view(tensor_x.size(0),1,
#                       tensor_x.size(2),
#                       tensor_x.size(3))

idx = torch.randperm(y.size(0))

x_sh = x[idx].view(x.size())
y_sh = y[idx].view(y.size())

x_sh = x_sh*2-1
y_sh = y_sh*2-1

del x,y, idx, data, tensor_y

batch_size = 64
# trainset = TensorDataset(x_sh[:-91],y_sh[:-91])
trainset = TensorDataset(x_sh[:-27],y_sh[:-27])

del x_sh,y_sh

trainloader = DataLoader(trainset, batch_size=batch_size,
                                          num_workers=42,
                                          pin_memory=True)

filt0= 64
filt1= 64
stker = 3
pd = 1
pm = 'replicate'
rd = 16

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

class DiffusionModel:
    def __init__(self, timesteps = 100):
        self.timesteps = timesteps
        
        """
        if 
            betas = [0.1, 0.2, 0.3, ...]
        then
            alphas = [0.9, 0.8, 0.7, ...]
            alphas_cumprod = [0.9, 0.9 * 0.8, 0.9 * 0.8, * 0.7, ...]
                    
        """ 
        self.betas = torch.from_numpy(cosine_beta_schedule(timesteps, s=0.008)).float()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat((torch.tensor([1.]), self.alphas_cumprod[:-1]),0)
        
    def forward(self, x_0, t, device):
        """
        x_0: (B, C, H, W)
        t: (B,)
        """
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.alphas_cumprod.sqrt(), t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x_0.shape)
            
        mean = sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)
        
        return mean + variance, noise.to(device)
    
    @torch.no_grad()
    def backward(self, x, t, cond, model, **kwargs):
        """
        Calls the model to predict the noise in the image and returns 
        the denoised image. 
        Applies noise to this image, if we are not in the last step yet.
        """
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        one_minus_alphas_cumprod_t = self.get_index_from_list(1. - self.alphas_cumprod, t, x.shape)
        one_minus_alphas_cumprod_prev_t = self.get_index_from_list(1. - self.alphas_cumprod_prev, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)
        mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, cond, **kwargs) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = betas_t * ((one_minus_alphas_cumprod_prev_t) / (one_minus_alphas_cumprod_t)) ** 0.5

        if t == 0:
            return mean
        else:
            noise = torch.randn_like(x)
            variance = torch.sqrt(posterior_variance_t) * noise 
            return mean + variance

    @staticmethod
    def get_index_from_list(values, t, x_shape):
        batch_size = t.shape[0]
        """
        pick the values from vals
        according to the indices stored in `t`
        """
        result = values.gather(-1, t.cpu())
        """
        if 
        x_shape = (5, 3, 64, 64)
            -> len(x_shape) = 4
            -> len(x_shape) - 1 = 3
            
        and thus we reshape `out` to dims
        (batch_size, 1, 1, 1)
        
        """
        return result.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings  

class convblock(nn.Module):

    def __init__(self, filt_in, filt_out):
        super(convblock, self).__init__()

        self.conv0 = nn.Conv2d(filt_in, filt_out, stker, padding=pd, padding_mode=pm)
        self.activation0 = nn.Mish(inplace=True)
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.activation0(x)

        return x

class finblock_Unet(nn.Module):

    def __init__(self, filt_in, filt_out):
        super(finblock_Unet, self).__init__()
        
        self.conv1 = nn.Conv2d(filt_in, 1, 1)
        
    def forward(self, x):
              
        x = self.conv1(x)

        return x

class CustPixelShuffle(nn.Sequential):
    def __init__(self, ni, nf, scale=2):
        super().__init__()
        layers = [nn.Conv2d(ni, nf*(scale**2), 1),
                  nn.PixelShuffle(scale)]        
        super().__init__(*layers)


class Block(nn.Module):
    def __init__(self, channels_in, channels_out, time_embedding_dims, num_filters = 3, downsample = True):
        super().__init__()
                
        self.downsample = downsample
              
        if downsample:
            self.conv1 = convblock(channels_in, channels_in)
            self.conv2 = convblock(channels_in, channels_in)
            self.conv3 = convblock(channels_in, channels_in)
            self.conv4 = convblock(channels_in, channels_in)           
            self.final = nn.Conv2d(channels_in, channels_out, 4, 2, 1)
        else:
            self.conv1 = convblock(channels_in, channels_in)
            self.conv2 = convblock(channels_in, channels_in)
            self.conv3 = convblock(channels_in, channels_in)
            self.conv4 = convblock(channels_in, channels_in)
            self.final = CustPixelShuffle(channels_in, channels_out)

            
        self.time_mlp = nn.Sequential(
                    nn.Mish(),
                    nn.Linear(time_embedding_dims, channels_in)
                )

    def forward(self, x, t, **kwargs):
        o = self.conv1(x)
        o_time = self.time_mlp(t)
        o = o + o_time[(..., ) + (None, ) * 2] 
        o = self.conv2(o)
        o += x
        x0 = o
        o = self.conv3(x)
        o_time0 = self.time_mlp(t)
        o = o + o_time0[(..., ) + (None, ) * 2] 
        o = self.conv4(o)
        o += x0        

        return self.final(o)

class MidBlock(nn.Module):
    def __init__(self, channels_in, channels_out, time_embedding_dims, num_filters = 3):
        super().__init__()
        
        self.conv1 = convblock(channels_in, channels_out)
        self.conv2 = convblock(channels_in, channels_out)
        self.conv3 = convblock(channels_in, channels_out)
        self.conv4 = convblock(channels_in, channels_out)

        self.time_mlp = nn.Sequential(
                    nn.Mish(),
                    nn.Linear(time_embedding_dims, channels_in)
                )
        
    def forward(self, x, t, **kwargs):
        o = self.conv1(x)
        o_time = self.time_mlp(t)
        o = o + o_time[(..., ) + (None, ) * 2] 
        o = self.conv2(o)
        o += x
        x0 = o
        o = self.conv3(x)
        o_time0 = self.time_mlp(t)
        o = o + o_time0[(..., ) + (None, ) * 2] 
        o = self.conv4(o)
        o += x0        

        return o

class UNet(nn.Module):
    def __init__(self, img_channels = 1, time_embedding_dims = 128):
        super().__init__()
        
        self.time_pos_emb = SinusoidalPositionEmbeddings(time_embedding_dims)
        self.mlp = nn.Sequential(
            nn.Linear(time_embedding_dims, time_embedding_dims * 4),
            nn.Mish(),
            nn.Linear(time_embedding_dims * 4, time_embedding_dims)
            )
        
        self.conv1 = nn.Conv2d(img_channels, filt1, stker, padding=pd, padding_mode=pm)
        self.conv2 = finblock_Unet(filt1, filt1)

        self.cond_proj = nn.ConvTranspose2d(filt0, filt1, 6, 6)
    
        self.downblock0 = Block(filt1,filt1*2,time_embedding_dims) 
        self.downblock1 = Block(filt1*2,filt1*4,time_embedding_dims)
        self.downblock2 = Block(filt1*4,filt1*8,time_embedding_dims)
        self.downblock3 = Block(filt1*8,filt1*8,time_embedding_dims) 
        self.midblock = MidBlock(filt1*8,filt1*8,time_embedding_dims)        
        self.upblock0 = Block(filt1*8*2,filt1*8,time_embedding_dims, downsample=False)
        self.upblock1 = Block(filt1*8*2,filt1*4,time_embedding_dims, downsample=False)
        self.upblock2 = Block(filt1*4*2,filt1*2,time_embedding_dims, downsample=False)
        self.upblock3 = Block(filt1*2*2,filt1,time_embedding_dims, downsample=False)  
               
    def forward(self, x, t, cond):
        t = self.time_pos_emb(t)
        t = self.mlp(t)
        
        o = self.conv1(x)
        cond0 = self.cond_proj(cond)
        o += cond0
        o = self.downblock0(o,t)
        res0 = o 
        o = self.downblock1(o,t)
        res1 = o 
        o = self.downblock2(o,t)
        res2 = o 
        o = self.downblock3(o,t)
        res3 = o 
        o = self.midblock(o, t)
        o = torch.cat((o, res3), dim=1)        
        o = self.upblock0(o,t)
        o = torch.cat((o, res2), dim=1)
        o = self.upblock1(o,t)
        o = torch.cat((o, res1), dim=1)
        o = self.upblock2(o,t)
        o = torch.cat((o, res0), dim=1) 
        o = self.upblock3(o,t)
        o = self.conv2(o)
        
        return o

# LR encoder ==================================================================

NEG_INF = -1000000

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Mish, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ARTTransformerBlock(nn.Module):
    r""" ART Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size: window size of dense attention
        interval: interval size of sparse attention
        ds_flag (int): use Dense Attention or Sparse Attention, 0 for DAB and 1 for SAB.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 interval=8,
                 ds_flag=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.interval = interval
        self.ds_flag = ds_flag
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size %d, %d, %d" % (L, H, W)

        if min(H, W) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.ds_flag = 0
            self.window_size = min(H, W)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # padding
        size_par = self.interval if self.ds_flag == 1 else self.window_size
        pad_l = pad_t = 0
        pad_r = (size_par - W % size_par) % size_par
        pad_b = (size_par - H % size_par) % size_par
        x = f.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hd, Wd, _ = x.shape

        mask = torch.zeros((1, Hd, Wd, 1), device=x.device)
        if pad_b > 0:
            mask[:, -pad_b:, :, :] = -1
        if pad_r > 0:
            mask[:, :, -pad_r:, :] = -1

        # partition the whole feature map into several groups
        if self.ds_flag == 0:  # Dense Attention
            G = Gh = Gw = self.window_size
            x = x.reshape(B, Hd // G, G, Wd // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.reshape(B * Hd * Wd // G ** 2, G ** 2, C)
            nP = Hd * Wd // G ** 2 # number of partitioning groups
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Hd // G, G, Wd // G, G, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
                mask = mask.reshape(nP, 1, G * G)
                attn_mask = torch.zeros((nP, G * G, G * G), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None
        if self.ds_flag == 1: # Sparse Attention
            I, Gh, Gw = self.interval, Hd // self.interval, Wd // self.interval
            x = x.reshape(B, Gh, I, Gw, I, C).permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.reshape(B * I * I, Gh * Gw, C)
            nP = I ** 2  # number of partitioning groups
            # attn_mask
            if pad_r > 0 or pad_b > 0:
                mask = mask.reshape(1, Gh, I, Gw, I, 1).permute(0, 2, 4, 1, 3, 5).contiguous()
                mask = mask.reshape(nP, 1, Gh * Gw)
                attn_mask = torch.zeros((nP, Gh * Gw, Gh * Gw), device=x.device)
                attn_mask = attn_mask.masked_fill(mask < 0, NEG_INF)
            else:
                attn_mask = None

        # MSA
        x = self.attn(x, Gh, Gw, mask=attn_mask)  # nP*B, Gh*Gw, C

        # merge embeddings
        if self.ds_flag == 0:
            x = x.reshape(B, Hd // G, Wd // G, G, G, C).permute(0, 1, 3, 2, 4,
                                                                5).contiguous()  # B, Hd//G, G, Wd//G, G, C
        else:
            x = x.reshape(B, I, I, Gh, Gw, C).permute(0, 3, 1, 4, 2, 5).contiguous()  # B, Gh, I, Gw, I, C
        x = x.reshape(B, Hd, Wd, C)

        # remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class BasicLayer(nn.Module):
    """ A basic ART Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): dense window size.
        interval: sparse interval size
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 interval,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            ds_flag = 0 if (i % 2 == 0) else 1
            self.blocks.append(ARTTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                interval=interval,
                ds_flag=ds_flag,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

class ResidualGroup(nn.Module):
    """Residual group including some ART Transformer Blocks (ResidualGroup).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): dense window size.
        interval: sparse interval size
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 interval,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv'):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            interval=interval,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        # build the last conv layer in each residual group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence
    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence
    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x


class ART(nn.Module):
    r""" ART
        A PyTorch impl of : `Accurate Image Restoration with Attention Retractable Transformer`.
    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each ART residual group.
        num_heads (tuple(int)): Number of attention heads in different layers.
        interval(tuple(int)): Interval size of sparse attention in different residual groups
        window_size (int): Window size of dense attention. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 img_size=(24,24),
                 patch_size=1,
                 in_chans=1,
                 embed_dim= 96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 interval=(4, 4, 4, 4),
                 window_size=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 resi_connection='1conv',
                 **kwargs):
        super(ART, self).__init__()
        num_in_ch = in_chans
        num_feat = 64

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = convblock(num_in_ch, embed_dim)
        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Group including ART Transformer blocks (ResidualGroup)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                interval=interval[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups

        # ------------------------- restoration module ------------------------- #
        self.conv_after_body = convblock(embed_dim, embed_dim)
        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        self.conv_before_upsample1 = convblock(embed_dim, num_feat)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x
    def forward(self, x):
        
        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample1(x)
        
        x = torch.clamp(x,min=-1,max=1)
        
        return x

# if __name__ == '__main__':

in_tile_size = 24

encoder= ART(img_size=(in_tile_size, in_tile_size))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = encoder.to(device)

# LRenc = encoder
checkpointART = torch.load('C://Users//Dfran//.spyder-py3//6xLrEnc_checkpoint100.pth', map_location=torch.device('cpu'))['model_state_dict']

checkpointART = torch.load('6xLrEnc_checkpoint100.pth')['model_state_dict']
encoder.load_state_dict(checkpointART,strict=False)

# Epoch: 49 | Train Loss 0.04000339342809254
# Epoch: 74 | Train Loss 0.04239618236226356
# Epoch: 149 | Train Loss 0.04093327142615821

# Epoch: 199 | Train Loss 0.039736586829203564

criterion = nn.L1Loss()

unet = UNet()
unet = unet.to(device)

# checkpointDiff = torch.load('6xSrdiff_checkpoint50.pth')

# unet.load_state_dict(checkpointDiff['model_state_dict'])

diffusion_model = DiffusionModel()

optimizer = torch.optim.Adam(unet.parameters(), lr=8e-5)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# optimizer.load_state_dict(checkpointDiff['optimizer_state_dict'])
# optimizer.param_groups[0]['lr'] = lr

for epoch in range(200):
    mean_epoch_loss = []
    for param in encoder.parameters():
        param.requires_grad = False
    for i, data in enumerate(trainloader, 0):
        batch = data[0].to(device) 
        labels = data[1].to(device)
        batch_up = f.interpolate(batch, (int(batch.shape[2]*6) , 
                                                int(batch.shape[3]*6)), 
                                  mode='bicubic').to(device)
        x_r = (labels - batch_up).to(device)
        encoder.eval()
        cond = encoder(batch)
        t = torch.randint(0, diffusion_model.timesteps, (batch_size,)).long().to(device)
        batch_noisy, noise = diffusion_model.forward(x_r, t, device) 
        predicted_noise = unet(batch_noisy, t, cond.detach())

        optimizer.zero_grad()
        loss = criterion(noise, predicted_noise)
        mean_epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    scheduler.step()    
    print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss)}")
        
torch.save({
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, '6xSrdiff_200epochs.pth') 
