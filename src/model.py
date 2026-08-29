"""
Modified Lightweight SwinIR for 4-Channel Sentinel-2 Super-Resolution (x4).
Adapted from "SwinIR: Image Restoration Using Swin Transformer" (Liang et al., 2021).

Specialized for Sentinel-2 10m (B04-Red, B03-Green, B02-Blue, B08-NIR) -> 2.5m 4-band output.
Memory optimized for 4GB VRAM (embed_dim=60, depths=[4,4,4,4], num_heads=[4,4,4,4], window_size=8).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA / SW-MSA) with relative position bias. """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # Relative position index coordinates
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block. """

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0..window_size-1"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, f"Input feature size ({L}) does not match H*W ({H*W})"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature maps to multiples of window size if necessary
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        if self.input_resolution == (Hp, Wp):
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            # Dynamic attention mask if resolution changed
            if self.shift_size > 0:
                img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
                h_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
                mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
                dyn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                dyn_mask = dyn_mask.masked_fill(dyn_mask != 0, float(-100.0)).masked_fill(dyn_mask == 0, float(0.0))
                attn_windows = self.attn(x_windows, mask=dyn_mask)
            else:
                attn_windows = self.attn(x_windows, mask=None)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """ Residual Swin Transformer Block (RSTB). """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):
        res = x
        for blk in self.blocks:
            res = blk(res, x_size)
        
        # Reshape to 2D image for conv
        B, L, C = res.shape
        H, W = x_size
        res = res.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        res = self.conv(res)
        res = res.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        return x + res


class Upsample(nn.Sequential):
    """ Upsample module using PixelShuffle for x4 scaling. """
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
                m.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        else:
            raise ValueError(f"scale {scale} is not supported. Supported scale is 2 or 4.")
        super().__init__(*m)


class SwinIRSatellite(nn.Module):
    """
    Lightweight SwinIR tailored for 4-channel Sentinel-2 (RGBN) Super-Resolution.
    Input: [B, 4, H, W] (10m)
    Output: [B, 4, 4H, 4W] (2.5m)
    """

    def __init__(self,
                 img_size=64,
                 in_chans=4,
                 out_chans=4,
                 upscale=4,
                 embed_dim=60,
                 depths=(4, 4, 4, 4),
                 num_heads=(4, 4, 4, 4),
                 window_size=8,
                 mlp_ratio=2.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.upscale = upscale
        self.embed_dim = embed_dim
        self.num_layers = len(depths)

        # 1. Shallow feature extraction (Converts 4-channel RGBN to embed_dim)
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction (Residual Swin Transformer Blocks)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(img_size, img_size),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=nn.LayerNorm
            )
            self.layers.append(layer)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # 3. High quality reconstruction & x4 PixelShuffle upsampling
        self.upsample = Upsample(upscale, embed_dim)
        self.conv_last = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, 4, H, W] in range [0, 1] or normalized reflectance.
        Returns:
            out: [B, 4, 4H, 4W] Super-resolved 2.5m RGBN imagery.
        """
        assert x.dim() == 4, f"Expected 4D tensor [B, C, H, W], got shape {x.shape}"
        assert x.shape[1] == self.in_chans, f"Expected {self.in_chans} channels, got {x.shape[1]}"
        
        B, C, H, W = x.shape
        x_size = (H, W)

        # Shallow feature
        x_first = self.conv_first(x)  # [B, embed_dim, H, W]

        # Deep features via Swin Transformer
        feat = x_first.permute(0, 2, 3, 1).contiguous().view(B, H * W, self.embed_dim)  # [B, H*W, embed_dim]
        for layer in self.layers:
            feat = layer(feat, x_size)
        feat = self.norm(feat)
        feat = feat.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        # Residual connection
        feat = self.conv_after_body(feat) + x_first

        # Upsampling to 2.5m
        feat = self.upsample(feat)
        out = self.conv_last(feat)

        return out


class ResidualSwinIRSatellite(nn.Module):
    """
    Residual SwinIR for Satellite Imagery.
    Learns strictly the high-frequency residual: HR - Bicubic(LR).
    Initialized with zero weights on conv_last so step 0 prediction is exactly Bicubic(LR).
    """
    def __init__(self,
                 img_size=64,
                 in_chans=4,
                 out_chans=4,
                 upscale=4,
                 embed_dim=60,
                 depths=(4, 4, 4, 4),
                 num_heads=(4, 4, 4, 4),
                 window_size=8,
                 mlp_ratio=2.0):
        super().__init__()
        self.upscale = upscale
        self.in_chans = in_chans
        self.out_chans = out_chans

        # Backbone SwinIR for residual extraction
        self.swin = create_swinir_satellite(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size
        )

        # Initialize residual head to zeros so at epoch 0: pred = bicubic + 0 = bicubic
        nn.init.zeros_(self.swin.conv_last.weight)
        if self.swin.conv_last.bias is not None:
            nn.init.zeros_(self.swin.conv_last.bias)

    def forward(self, x):
        # 1. Bicubic base interpolation
        bicubic_base = F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        # 2. Learned high-frequency residual
        residual = self.swin(x)
        # 3. Combined output
        return bicubic_base + residual


def create_swinir_satellite(embed_dim=60, depths=(4, 4, 4, 4), num_heads=(4, 4, 4, 4), window_size=8):
    """
    Factory function for Phase 1 direct lightweight SwinIR.
    """
    return SwinIRSatellite(
        img_size=64,
        in_chans=4,
        out_chans=4,
        upscale=4,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=2.0
    )


def create_residual_swinir_satellite(embed_dim=60, depths=(4, 4, 4, 4), num_heads=(4, 4, 4, 4), window_size=8):
    """
    Factory function for Phase 1 Residual SwinIR (Bicubic + Residual).
    """
    return ResidualSwinIRSatellite(
        img_size=64,
        in_chans=4,
        out_chans=4,
        upscale=4,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=2.0
    )


if __name__ == "__main__":
    model = create_residual_swinir_satellite()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Residual-SwinIR-Satellite initialized successfully. Total trainable parameters: {num_params:,}")
    
    # Test forward pass with batch size 2 and 64x64 input
    dummy_lr = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        out_hr = model(dummy_lr)
    print(f"Input shape: {dummy_lr.shape} -> Output shape: {out_hr.shape}")
    assert out_hr.shape == (2, 4, 256, 256), "Shape mismatch!"
    print("Self-test passed!")
