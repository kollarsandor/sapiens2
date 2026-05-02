import math
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import trunc_normal_
from torch.utils.checkpoint import checkpoint


def to_2tuple(x):
    if isinstance(x, (str, bytes)):
        return (x, x)
    if isinstance(x, Sequence):
        x = tuple(x)
        if len(x) == 2:
            return x
        raise ValueError("Expected scalar or length-2 iterable")
    return (x, x)


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: Optional[float] = 100.0,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % (4 * num_heads) != 0:
            raise ValueError("embed_dim must be divisible by 4 * num_heads")
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError(
                "Either `base` or `min_period`+`max_period` must be provided."
            )
        if base is not None and base <= 0:
            raise ValueError("base must be positive")
        if both_periods:
            if min_period is None or max_period is None:
                raise ValueError("min_period and max_period must both be provided")
            if min_period <= 0 or max_period <= 0:
                raise ValueError("min_period and max_period must be positive")
            if max_period < min_period:
                raise ValueError("max_period must be greater than or equal to min_period")
        if normalize_coords not in ("min", "max", "separate"):
            raise ValueError(f"Unknown normalize_coords: {normalize_coords}")
        if shift_coords is not None and shift_coords < 0:
            raise ValueError("shift_coords must be non-negative")
        if jitter_coords is not None and jitter_coords < 1:
            raise ValueError("jitter_coords must be greater than or equal to 1")
        if rescale_coords is not None and rescale_coords < 1:
            raise ValueError("rescale_coords must be greater than or equal to 1")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype or torch.float32
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=self.dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> Tuple[Tensor, Tensor]:
        if H <= 0 or W <= 0:
            raise ValueError("H and W must be positive")
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW
            coords_w = torch.arange(0.5, W, **dd) / max_HW
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW
            coords_w = torch.arange(0.5, W, **dd) / min_HW
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        )
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(
                -self.shift_coords, self.shift_coords
            )
            coords = coords + shift_hw[None, :]

        if self.training and self.jitter_coords is not None:
            jitter_max = math.log(float(self.jitter_coords))
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords = coords * jitter_hw[None, :]

        if self.training and self.rescale_coords is not None:
            rescale_max = math.log(float(self.rescale_coords))
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords = coords * rescale_hw

        angles = (
            2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        )
        angles = angles.flatten(1, 2)
        angles = angles.repeat(1, 2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return sin, cos

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2
                * torch.arange(self.D_head // 4, device=device, dtype=dtype)
                / (self.D_head // 2)
            )
        else:
            if self.min_period is None or self.max_period is None:
                raise ValueError("min_period and max_period must both be provided")
            base = self.max_period / self.min_period
            exponents = torch.linspace(
                0, 1, self.D_head // 4, device=device, dtype=dtype
            )
            periods = self.min_period * (base**exponents)
        with torch.no_grad():
            self.periods.copy_(periods)


class Tokenizer(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        window_size: int = 4,
        num_heads: int = 4,
        num_tokenizer_layers: int = 1,
        qkv_bias: bool = True,
        use_qk_norm: bool = False,
        chunk_size: int = 1024,
    ):
        super().__init__()
        if embed_dims <= 0:
            raise ValueError("embed_dims must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if num_tokenizer_layers < 0:
            raise ValueError("num_tokenizer_layers must be non-negative")
        self.ws = window_size
        self.chunk_size = chunk_size

        self.local_pos_embed = nn.Parameter(
            torch.zeros(1, 1 + window_size * window_size, embed_dims)
        )
        trunc_normal_(self.local_pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer2(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=embed_dims * 4,
                    qkv_bias=qkv_bias,
                    use_qk_norm=use_qk_norm,
                )
                for _ in range(num_tokenizer_layers)
            ]
        )

        self.w_cls = nn.Parameter(torch.zeros(1, 1, embed_dims))
        trunc_normal_(self.w_cls, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x.dim() != 3:
            raise ValueError("x must have shape (B, N, C)")
        B, N, C = x.shape
        H, W = hw
        ws = self.ws
        if H <= 0 or W <= 0:
            raise ValueError("H and W must be positive")
        if N != H * W:
            raise ValueError("N must equal H * W")
        if H % ws != 0 or W % ws != 0:
            raise ValueError(f"Image size {H}×{W} must be divisible by window {ws}.")

        x = x.view(B, H, W, C)

        ph, pw = H // ws, W // ws
        ph, pw = int(ph), int(pw)
        x = x.view(B, ph, ws, pw, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.contiguous().view(B * ph * pw, ws * ws, C)

        total_windows = x.size(0)
        chunk_size = int(min(self.chunk_size, total_windows))
        token_out = x.new_empty(total_windows, C)

        use_ckpt = self.training and torch.is_grad_enabled()

        def _run_blocks(t: torch.Tensor) -> torch.Tensor:
            for blk in self.blocks:
                t = blk(t)
            return t

        for i in range(0, total_windows, chunk_size):
            chunk = x[i : i + chunk_size]
            m = chunk.size(0)
            cls = self.w_cls.expand(m, -1, -1)
            chunk = torch.cat([cls, chunk], dim=1)
            chunk = chunk + self.local_pos_embed

            if use_ckpt:
                chunk = checkpoint(_run_blocks, chunk, use_reentrant=False)
            else:
                chunk = _run_blocks(chunk)

            token_out[i : i + m] = chunk[:, 0]

        token = token_out.view(B, ph * pw, C)
        return token, (ph, pw)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        num_kv_heads=None,
        input_dims=None,
        attn_drop=0.0,
        proj_drop=0.0,
        qkv_bias=True,
        qk_scale=None,
        proj_bias=True,
        use_qk_norm=True,
        v_shortcut=False,
        layer_scale_init_value=0.0,
    ):
        super().__init__()
        if embed_dims <= 0:
            raise ValueError("embed_dims must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dims % num_heads != 0:
            raise ValueError("embed_dims must be divisible by num_heads")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_kv_heads must divide num_heads")
        self.head_dim = embed_dims // num_heads
        self.input_dims = input_dims if input_dims is not None else embed_dims
        if self.input_dims <= 0:
            raise ValueError("input_dims must be positive")
        if not 0.0 <= attn_drop <= 1.0:
            raise ValueError("attn_drop must be between 0 and 1")
        if not 0.0 <= proj_drop <= 1.0:
            raise ValueError("proj_drop must be between 0 and 1")
        if qk_scale is not None and qk_scale <= 0:
            raise ValueError("qk_scale must be positive")
        self.attn_drop = float(attn_drop)
        self.qk_scale = qk_scale
        self.v_shortcut = v_shortcut
        self.use_qk_norm = use_qk_norm
        self.attn_op = F.scaled_dot_product_attention

        self.wq = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self.wk = nn.Linear(
            self.input_dims, self.num_kv_heads * self.head_dim, bias=qkv_bias
        )
        self.wv = nn.Linear(
            self.input_dims, self.num_kv_heads * self.head_dim, bias=qkv_bias
        )

        if self.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if layer_scale_init_value > 0:
            self.gamma = LayerScale(embed_dims, scale=layer_scale_init_value)
        else:
            self.gamma = nn.Identity()

    def apply_rope(
        self, q: Tensor, k: Tensor, rope: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        if len(rope) != 2:
            raise ValueError("rope must contain sin and cos tensors")
        sin, cos = rope
        if sin.shape != cos.shape:
            raise ValueError("sin and cos must have the same shape")
        if q.shape[-1] != k.shape[-1]:
            raise ValueError("q and k must have the same head dimension")
        if q.shape[-1] != sin.shape[-1]:
            raise ValueError("rope head dimension must match q and k head dimension")
        if q.shape[-1] % 2 != 0:
            raise ValueError("rope head dimension must be even")
        q_dtype = q.dtype
        k_dtype = k.dtype
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        rope_len = sin.shape[-2]
        q_prefix_len = q.shape[-2] - rope_len
        k_prefix_len = k.shape[-2] - rope_len
        if q_prefix_len < 0 or k_prefix_len < 0:
            raise ValueError("rope sequence length cannot exceed q or k sequence length")
        q_prefix = q[:, :, :q_prefix_len, :]
        q = self._rope_apply(q[:, :, q_prefix_len:, :], sin, cos)
        q = torch.cat((q_prefix, q), dim=-2)
        k_prefix = k[:, :, :k_prefix_len, :]
        k = self._rope_apply(k[:, :, k_prefix_len:, :], sin, cos)
        k = torch.cat((k_prefix, k), dim=-2)
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def _rope_rotate_half(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _rope_apply(self, x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        return (x * cos) + (self._rope_rotate_half(x) * sin)

    def forward(self, x, rope=None):
        if x.dim() != 3:
            raise ValueError("x must have shape (B, N, C)")
        B, N, C = x.shape
        if C != self.input_dims:
            raise ValueError("x last dimension must match input_dims")
        q = self.wq(x).view(B, N, self.num_heads, self.head_dim)
        k = self.wk(x).view(B, N, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(B, N, self.num_kv_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.num_kv_heads != self.num_heads:
            factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(factor, dim=1)
            v = v.repeat_interleave(factor, dim=1)

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        dropout_p = self.attn_drop if self.training else 0.0
        if self.qk_scale is None:
            attn_out = self.attn_op(q, k, v, dropout_p=dropout_p)
        else:
            attn_out = self.attn_op(q, k, v, dropout_p=dropout_p, scale=self.qk_scale)

        out = attn_out.permute(0, 2, 1, 3).reshape(B, N, self.embed_dims)

        if self.v_shortcut:
            v_out = v.permute(0, 2, 1, 3).reshape(B, N, self.embed_dims)
            out = out + v_out

        out = self.proj(out)
        out = self.gamma(self.proj_drop(out))

        return out


class TransformerEncoderLayer2(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        num_kv_heads=None,
        feedforward_channels=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_scale_init_value=0.0,
        use_qk_norm=True,
        qkv_bias=True,
    ):
        super(TransformerEncoderLayer2, self).__init__()

        self.embed_dims = embed_dims
        self.ln1 = nn.RMSNorm(self.embed_dims, eps=1e-6)
        self.attn = GroupedQueryAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            qkv_bias=qkv_bias,
            layer_scale_init_value=layer_scale_init_value,
            use_qk_norm=use_qk_norm,
        )

        self.ln2 = nn.RMSNorm(self.embed_dims, eps=1e-6)
        self.ffn = SwiGLUFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            layer_scale_init_value=layer_scale_init_value,
        )

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x, rope=None):
        x = x + self.attn(self.ln1(x), rope=rope)
        x = self.ffn(self.ln2(x), identity=x)
        return x


class Sapiens2(nn.Module):
    arch_zoo = {
        **dict.fromkeys(
            ["sapiens2_0.1b"],
            {
                "embed_dims": 768,
                "num_layers": 12,
                "num_heads": 12,
                "feedforward_channels": 768 * 4,
                "num_tokenizer_layers": 2,
            },
        ),
        **dict.fromkeys(
            ["sapiens2_0.4b"],
            {
                "embed_dims": 1024,
                "num_layers": 24,
                "num_heads": 16,
                "feedforward_channels": 1024 * 4,
                "num_tokenizer_layers": 2,
            },
        ),
        **dict.fromkeys(
            ["sapiens2_0.8b"],
            {
                "embed_dims": 1280,
                "num_layers": 32,
                "num_heads": 16,
                "feedforward_channels": 1280 * 4,
                "num_tokenizer_layers": 3,
            },
        ),
        **dict.fromkeys(
            ["sapiens2_1b"],
            {
                "embed_dims": 1536,
                "num_layers": 40,
                "num_heads": 24,
                "feedforward_channels": 1536 * 4,
                "num_tokenizer_layers": 4,
            },
        ),
        **dict.fromkeys(
            ["sapiens2_5b"],
            {
                "embed_dims": 2432,
                "num_layers": 56,
                "num_heads": 32,
                "feedforward_channels": 2432 * 4,
                "num_tokenizer_layers": 6,
            },
        ),
    }

    num_extra_tokens = 1
    OUT_TYPES = {"raw", "cls_token", "featmap"}

    def __init__(
        self,
        arch="sapiens2_1b",
        img_size=(1024, 768),
        patch_size=16,
        in_channels=3,
        out_indices=-1,
        drop_rate=0.0,
        window_size=4,
        use_tokenizer=False,
        use_qk_norm=True,
        qkv_bias=True,
        final_norm=True,
        out_type="raw",
        with_cls_token=True,
        layer_scale_init_value=1e-4,
        frozen_stages=-1,
        patch_cfg: Optional[Dict[str, Any]] = None,
        layer_cfgs: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
        pos_embed_rope_base: Optional[float] = 100.0,
        pos_embed_rope_min_period: Optional[float] = None,
        pos_embed_rope_max_period: Optional[float] = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: Optional[float] = None,
        pos_embed_rope_jitter_coords: Optional[float] = None,
        pos_embed_rope_rescale_coords: Optional[float] = None,
        pos_embed_rope_dtype: str = "bf16",
        n_storage_tokens: int = 8,
    ):
        super().__init__()

        if not isinstance(arch, str):
            raise TypeError("arch must be a string")
        arch = arch.lower()
        if arch not in set(self.arch_zoo):
            raise ValueError(f"Arch {arch} is not in default archs {set(self.arch_zoo)}")
        self.arch_settings = self.arch_zoo[arch]

        if window_size <= 0:
            raise ValueError("window_size must be positive")
        patch_size_tuple = to_2tuple(patch_size)
        if patch_size_tuple[0] <= 0 or patch_size_tuple[1] <= 0:
            raise ValueError("patch_size must be positive")
        img_size = to_2tuple(img_size)
        if img_size[0] <= 0 or img_size[1] <= 0:
            raise ValueError("img_size must be positive")
        if frozen_stages < -1 or frozen_stages > self.arch_settings["num_layers"]:
            raise ValueError("frozen_stages must be between -1 and num_layers")

        self.embed_dims = self.arch_settings["embed_dims"]
        self.num_layers = self.arch_settings["num_layers"]
        self.patch_size = patch_size
        self.window_size = window_size
        self.img_size = img_size

        patch_cfg = dict(patch_cfg or {})
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=self.img_size,
            embed_dims=self.embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        if self.patch_embed.init_out_size is None:
            raise ValueError("PatchEmbed requires a known input_size")
        patch_embed_resolution = self.patch_embed.init_out_size
        if use_tokenizer:
            if (
                patch_embed_resolution[0] % window_size != 0
                or patch_embed_resolution[1] % window_size != 0
            ):
                raise ValueError(
                    "Patch resolution must be divisible by window_size when use_tokenizer=True"
                )
            self.patch_resolution = (
                patch_embed_resolution[0] // window_size,
                patch_embed_resolution[1] // window_size,
            )
        else:
            self.patch_resolution = patch_embed_resolution

        if pos_embed_rope_dtype in ("bf16", "bfloat16"):
            rope_dtype = torch.bfloat16
        elif pos_embed_rope_dtype in ("fp32", "float32"):
            rope_dtype = torch.float32
        elif pos_embed_rope_dtype in (torch.bfloat16, torch.float32):
            rope_dtype = pos_embed_rope_dtype
        else:
            raise ValueError("pos_embed_rope_dtype must be 'bf16' or 'float32'")

        self.rope_embed = RopePositionEmbedding(
            embed_dim=self.embed_dims,
            num_heads=self.arch_settings["num_heads"],
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=rope_dtype,
        )

        if out_type not in self.OUT_TYPES:
            raise ValueError(
                f"Unsupported `out_type` {out_type}, please choose from {self.OUT_TYPES}"
            )
        self.out_type = out_type

        if use_tokenizer:
            self.tokenizer = Tokenizer(
                embed_dims=self.embed_dims,
                window_size=self.window_size,
                num_heads=self.arch_settings["num_heads"],
                num_tokenizer_layers=self.arch_settings["num_tokenizer_layers"],
                qkv_bias=qkv_bias,
                use_qk_norm=False,
            )
        else:
            self.tokenizer = None

        self.with_cls_token = with_cls_token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        elif out_type != "cls_token":
            self.cls_token = None
            self.num_extra_tokens = 0
        else:
            raise ValueError('with_cls_token must be True when `out_type="cls_token"`.')

        self.n_storage_tokens = int(n_storage_tokens)
        if self.n_storage_tokens < 0:
            raise ValueError("n_storage_tokens must be non-negative")
        self.storage_tokens = (
            nn.Parameter(torch.zeros(1, self.n_storage_tokens, self.embed_dims))
            if self.n_storage_tokens > 0
            else None
        )
        self.num_extra_tokens = (
            1 if self.cls_token is not None else 0
        ) + self.n_storage_tokens

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        elif isinstance(out_indices, Sequence) and not isinstance(out_indices, (str, bytes)):
            out_indices = list(out_indices)
        else:
            raise TypeError(
                f'"out_indices" must be a sequence or int, got {type(out_indices)} instead.'
            )
        normalized_out_indices = []
        for index in out_indices:
            if not isinstance(index, int):
                raise TypeError("All out_indices entries must be integers")
            original_index = index
            if index < 0:
                index = self.num_layers + index
            if not 0 <= index < self.num_layers:
                raise ValueError(f"Invalid out_indices {original_index}")
            normalized_out_indices.append(index)
        self.out_indices = tuple(normalized_out_indices)

        self.blocks = nn.ModuleList()
        if layer_cfgs is None:
            layer_cfgs = {}
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [dict(layer_cfgs) for _ in range(self.num_layers)]
        else:
            if not isinstance(layer_cfgs, Sequence) or isinstance(layer_cfgs, (str, bytes)):
                raise TypeError("layer_cfgs must be a dict or a sequence of dicts")
            if len(layer_cfgs) != self.num_layers:
                raise ValueError("layer_cfgs length must match num_layers")
            layer_cfgs = [dict(layer_cfg or {}) for layer_cfg in layer_cfgs]

        mhsa_early, mhsa_late = 8, 8
        for i in range(self.num_layers):
            if i < mhsa_early or i >= self.num_layers - mhsa_late:
                num_kv_heads = None
            else:
                num_kv_heads = self.arch_settings["num_heads"] // 2

            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings["num_heads"],
                num_kv_heads=num_kv_heads,
                feedforward_channels=self.arch_settings["feedforward_channels"],
                use_qk_norm=use_qk_norm,
                layer_scale_init_value=layer_scale_init_value,
                drop_rate=drop_rate,
                qkv_bias=qkv_bias,
            )
            _layer_cfg.update(layer_cfgs[i])
            self.blocks.append(TransformerEncoderLayer2(**_layer_cfg))

        self.frozen_stages = frozen_stages
        self.final_norm = final_norm
        self.ln1 = nn.RMSNorm(self.embed_dims, eps=1e-6) if final_norm else nn.Identity()

        self.init_weights()

        if self.frozen_stages > 0:
            self._freeze_stages()

    def init_weights(self):
        if self.with_cls_token:
            trunc_normal_(self.cls_token, std=0.02)

        if self.storage_tokens is not None:
            trunc_normal_(self.storage_tokens, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _freeze_stages(self):
        if self.frozen_stages >= 1 and self.tokenizer is not None:
            self.tokenizer.eval()
            for param in self.tokenizer.parameters():
                param.requires_grad = False

        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        if self.cls_token is not None:
            self.cls_token.requires_grad_(False)
        if self.storage_tokens is not None:
            self.storage_tokens.requires_grad_(False)

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.frozen_stages == len(self.blocks):
            if self.final_norm:
                self.ln1.eval()
                for param in self.ln1.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.frozen_stages > 0:
            self._freeze_stages()
        return self

    def forward(self, x):
        B = x.shape[0]

        x, patch_resolution = self.patch_embed(x)
        if self.tokenizer is not None:
            x, patch_resolution = self.tokenizer(x, patch_resolution)

        prepend = []
        if self.cls_token is not None:
            prepend.append(self.cls_token.expand(B, -1, -1))
        if self.storage_tokens is not None:
            prepend.append(self.storage_tokens.expand(B, -1, -1))
        if len(prepend) > 0:
            x = torch.cat(prepend + [x], dim=1)

        rope_sincos = self.rope_embed(H=patch_resolution[0], W=patch_resolution[1])
        outs = []
        for i, layer in enumerate(self.blocks):
            x = layer(x, rope=rope_sincos)

            if i == len(self.blocks) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)

    def _format_output(self, x, hw):
        if self.out_type == "raw":
            return x
        if self.out_type == "cls_token":
            return x[:, 0]

        patch_token = x[:, self.num_extra_tokens :]
        if self.out_type == "featmap":
            B = x.size(0)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        raise RuntimeError(f"Unsupported out_type {self.out_type}")

    @property
    def norm1(self):
        return self.ln1


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        inplace: bool = False,
        data_format: str = "channels_last",
        scale: float = 1e-5,
    ):
        super().__init__()
        if data_format not in (
            "channels_last",
            "channels_first",
        ):
            raise ValueError("'data_format' could only be channels_last or channels_first.")
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * scale)

    def forward(self, x) -> torch.Tensor:
        if self.data_format == "channels_first":
            shape = tuple((1, -1, *(1 for _ in range(x.dim() - 2))))
        else:
            shape = tuple((*(1 for _ in range(x.dim() - 1)), -1))
        if self.inplace:
            return x.mul_(self.weight.view(*shape))
        else:
            return x * self.weight.view(*shape)


class AdaptivePadding(nn.Module):
    def __init__(self, kernel_size=1, stride=1, dilation=1, padding="corner"):
        super().__init__()
        if padding not in ("same", "corner"):
            raise ValueError("padding must be 'same' or 'corner'")
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)
        self.padding = padding

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        dilation_h, dilation_w = self.dilation
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * dilation_h + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * dilation_w + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.shape[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = F.pad(x, [0, pad_w, 0, pad_h])
            else:
                x = F.pad(
                    x,
                    [
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                    ],
                )
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dims=768,
        kernel_size=16,
        stride=16,
        padding="corner",
        dilation=1,
        bias=True,
        input_size=None,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            )
            conv_padding = (0, 0)
        else:
            self.adaptive_padding = None
            conv_padding = to_2tuple(padding)

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            bias=bias,
        )

        if input_size is not None:
            input_size = to_2tuple(input_size)
            self.init_input_size = input_size
            if self.adaptive_padding is not None:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(input_size)
                h_out = (
                    input_size[0] + pad_h - dilation[0] * (kernel_size[0] - 1) - 1
                ) // stride[0] + 1
                w_out = (
                    input_size[1] + pad_w - dilation[1] * (kernel_size[1] - 1) - 1
                ) // stride[1] + 1
            else:
                h_out = (
                    input_size[0]
                    + 2 * conv_padding[0]
                    - dilation[0] * (kernel_size[0] - 1)
                    - 1
                ) // stride[0] + 1
                w_out = (
                    input_size[1]
                    + 2 * conv_padding[1]
                    - dilation[1] * (kernel_size[1] - 1)
                    - 1
                ) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        if self.adaptive_padding is not None:
            x = self.adaptive_padding(x)
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, out_size


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        out_dims: Optional[int] = None,
        layer_scale_init_value: float = 0.0,
        bias: bool = True,
        add_identity: bool = True,
    ) -> None:
        super().__init__()
        if embed_dims <= 0:
            raise ValueError("embed_dims must be positive")
        if feedforward_channels is not None and feedforward_channels <= 0:
            raise ValueError("feedforward_channels must be positive")
        if out_dims is not None and out_dims <= 0:
            raise ValueError("out_dims must be positive")
        self.embed_dims = embed_dims
        self.out_dims = out_dims or embed_dims
        hidden_dims = feedforward_channels or embed_dims

        self.w12 = nn.Linear(self.embed_dims, 2 * hidden_dims, bias=bias)
        self.w3 = nn.Linear(hidden_dims, self.out_dims, bias=bias)

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(dim=self.out_dims, scale=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

        self.add_identity = add_identity

    def forward(
        self, x: torch.Tensor, identity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        out = self.w3(hidden)
        out = self.gamma2(out)

        if self.out_dims != self.embed_dims or not self.add_identity:
            return out

        if identity is None:
            identity = x
        return identity + out
