from typing import Dict, List, Tuple, Union, Optional, Type, Callable, Any
from inspect import signature
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from einops import rearrange
# from .softpool import SoftPool2d

__all__ = [
    "efficientvit_b0",
    "efficientvit_b1",
    "efficientvit_b2",
    "efficientvit_b3",
]


#################################################################################
#                             Basic Layers                                      #
#################################################################################

def build_kwargs_from_config(config: Dict, target_func: Callable) -> Dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


REGISTERED_NORM_DICT: Dict[str, Type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name == "ln":
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None

class ExpActivation(nn.Module):
    def __init__(self):
        super(ExpActivation, self).__init__()

    def forward(self, input):
        out = torch.exp(input/10)
        return out
class Eluplus1(nn.Module):
    def __init__(self):
        super(Eluplus1, self).__init__()

    def forward(self, input):
        out = F.elu(input)+1.
        return out

REGISTERED_ACT_DICT: Dict[str, Type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "sigmoid": nn.Sigmoid,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "exp":ExpActivation,
    "elup1": Eluplus1,
}


def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None


def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def list_sum(x: List) -> Any:
    return x[0] if len(x) == 1 else x[0] + list_sum(x[1:])


def merge_tensor(x: List[torch.Tensor], mode="cat", dim=1) -> torch.Tensor:
    if mode == "cat":
        return torch.cat(x, dim=dim)
    elif mode == "add":
        return list_sum(x)
    else:
        raise NotImplementedError


def resize(
        x: torch.Tensor,
        size: Optional[Any] = None,
        scale_factor: Optional[List[float]] = None,
        mode: str = "bicubic",
        align_corners: Optional[bool] = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


def val2list(x: Union[List, Tuple, Any], repeat_time=1) -> List:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: Union[List, Tuple, Any], min_len: int = 1, idx_repeat: int = -1) -> Tuple:
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


class ConvLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
            dilation=1,
            groups=1,
            padding=None,
            use_bias=False,
            dropout_rate=0.2,
            norm="bn2d",
            act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.dropout = nn.Dropout2d(dropout_rate, inplace=False) if dropout_rate > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    def __init__(
            self,
            mode="bicubic",
            size: Union[int, Tuple[int, int], List[int], None] = None,
            factor=2,
            align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class LinearLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            use_bias=True,
            dropout_rate=0.2,
            norm=None,
            act_func=None,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout_rate, inplace=False) if dropout_rate > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x



#################################################################################
#                             Basic Blocks                                      #
#################################################################################


class DSConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
            use_bias=False,
            norm=("bn2d", "bn2d"),
            act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
            mid_channels=None,
            expand_ratio=6,
            use_bias=False,
            norm=("bn2d", "bn2d", "bn2d"),
            act_func=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class LiteMSA(nn.Module):
    r""" Lightweight multi-scale attention """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: Optional[int] = None,
            heads_ratio: float = 1.0,
            dim=8,
            use_bias=False,
            norm=(None, "bn2d"),
            act_func=(None, None),
            kernel_func="relu",
            scales: Tuple[int, ...] = (5,),
    ):
        super(LiteMSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim, 3 * total_dim, scale, padding=get_same_padding(scale), groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),

                    nn.Conv2d(
                        3 * total_dim, 3 * total_dim, 1, groups=3 * total_dim,
                        bias=use_bias[0],),
                )
                for scale in scales
            ]
        )



        self.kernel_func = build_act(kernel_func, inplace=False)# False

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())

        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        multi_scale_qkv = torch.reshape(
            multi_scale_qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
        q, k, v = (
            multi_scale_qkv[..., 0: self.dim].clone(),
            multi_scale_qkv[..., self.dim: 2 * self.dim].clone(),
            multi_scale_qkv[..., 2 * self.dim:].clone(),
        )

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

        # final projecttion
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        out = self.proj(out)
        return out


class LiteMCA(nn.Module):
    r""" Lightweight multi-scale attention """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: Optional[int] = None,
            heads_ratio: float = 1.0,
            dim=8,
            use_bias=False,
            norm=(None, "bn2d"),
            act_func=(None, None),
            kernel_func="relu",
            scales: Tuple[int, ...] = (5,),
    ):
        super(LiteMCA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.q = ConvLayer(
            in_channels,
            total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.kv = nn.Sequential(ConvLayer(
            in_channels,
            2 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        total_dim, total_dim, scale, padding=get_same_padding(scale), groups=total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(total_dim, total_dim, 1, groups=heads, bias=use_bias[0]),

                    nn.Conv2d(
                        total_dim, total_dim, 1, groups=total_dim,
                        bias=use_bias[0],),
                )
                for scale in scales
            ]
        )
        self.aggreg2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        2*total_dim, 2*total_dim, scale, padding=get_same_padding(scale), groups=2*total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(2*total_dim, 2*total_dim, 1, groups=2*heads, bias=use_bias[0]),

                    nn.Conv2d(
                        2*total_dim, 2*total_dim, 1, groups=2*total_dim,
                        bias=use_bias[0], ),
                )
                for scale in scales
            ]
        )




        self.kernel_func = build_act(kernel_func, inplace=False)# False

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor,y: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())

        # generate multi-scale q, k, v
        q = self.q(x)
        kv = self.kv(y)
        # qkv = torch.cat([q,kv],dim=1)
        multi_scale_q = [q]
        multi_scale_kv = [kv]
        for op in self.aggreg:
            multi_scale_q.append(op(q))
        for op in self.aggreg2:
            multi_scale_kv.append(op(kv))
        multi_scale_q = torch.cat(multi_scale_q, dim=1)
        multi_scale_kv = torch.cat(multi_scale_kv, dim=1)

        multi_scale_q = torch.reshape(
            multi_scale_q,
            (
                B,
                -1,
                self.dim,
                H * W,
            ),
        )
        kv = torch.reshape(
            multi_scale_kv,
            (
                B,
                -1,
                2 * self.dim,
                1,
            ),
        )

        q = torch.transpose(multi_scale_q, -1, -2)
        kv = torch.transpose(kv, -1, -2)

        k = kv[..., 0:self.dim]
        v = kv[..., self.dim:2*self.dim]

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

        # final projecttion
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        out = self.proj(out)
        return out

def bhwc2bchw( x):
    # (B, H * W, C) --> (B, C, H, W)
    B,S,HW, C = x.shape
    H = int(HW ** (0.5))
    x = x.reshape(-1, H, H, C).permute(0, 3, 1, 2)
    return x,[B,S,HW, C]
def bchw2bhwc( x,S):
    x = x.reshape(S[0], S[1],S[3], S[2]).permute(0,1, 3, 2)
    return x
class PoolMSA(nn.Module):
    r""" Lightweight multi-scale attention """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: Optional[int] = None,
            heads_ratio: float = 1.0,
            dim=8,
            use_bias=False,
            norm=(None, "bn2d"),
            act_func=(None, None),
            kernel_func="relu",
            scales: Tuple[int, ...] = (5,),
    ):
        super(PoolMSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim

        self.pool16x = SoftPool2d(17,16,1)
        self.conv16x = nn.Conv2d(in_channels=self.dim*2,out_channels=self.dim*2,kernel_size=17,stride=16,padding=1,groups = self.dim*2)


        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AvgPool2d(
                        kernel_size=scale, stride=1,padding=get_same_padding(scale)
                        ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                    nn.BatchNorm2d(3 * total_dim),
                )
                for scale in scales
            ]
        )



        self.kernel_func = nn.Sequential(build_act(kernel_func, inplace=False)
                                         )# False

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())

        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        multi_scale_qkv = torch.reshape(
            multi_scale_qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
        q, kv = (
            multi_scale_qkv[..., 0: self.dim].clone(),
            multi_scale_qkv[..., self.dim:].clone(),
        )
        kv = torch.transpose(kv, -1, -2)
        kv = kv.reshape(-1, 2 * self.dim, H, W)
        kv = self.conv16x(kv)
        # kv = F.interpolate(kv,size = [H//16,W//16],mode='bilinear')

        kv = kv.reshape(B,-1, 2 * self.dim, H//16 * W//16).transpose(-1,-2)
        k = kv[..., 0: self.dim].clone()
        v = kv[..., self.dim:].clone()
        # q = torch.transpose(q,-1,-2).reshape(-1,self.dim,H,W)
        # k = torch.transpose(k,-1,-2).reshape(-1,self.dim,H,W)
        # lightweight global attention
        # q,S = bhwc2bchw(q)
        # k,_ = bhwc2bchw(k)
        # q = bchw2bhwc(self.kernel_func(q),S)
        # k = bchw2bhwc(self.kernel_func(k),S)
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        # q = q.reshape(B,-1,self.dim,H*W).transpose(-1,-2)
        # k = k.reshape(B,-1,self.dim,H*W).transpose(-1,-2)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

        # final projecttion
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        out = self.proj(out)
        return out

class LiteTopkMSA(nn.Module):
    r""" Lightweight multi-scale attention """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: Optional[int] = None,
            heads_ratio: float = 1.0,
            dim=8,
            use_bias=False,
            norm=(None, "bn2d"),
            act_func=(None, None),
            kernel_func="relu",
            scales: Tuple[int, ...] = (5,),
    ):
        super(LiteTopkMSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            3,
            stride=1,
            padding=1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim, 3 * total_dim, scale, padding=get_same_padding(scale), groups=3 * total_dim,
                        bias=use_bias[0],
                        ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )

        self.embed = ConvLayer(
                                in_channels,
                                1,
                                8,
                                stride=8,
                                padding=0,
                                use_bias=use_bias[0],
                                norm=norm[0],
                                act_func='sigmoid',
                                )

        self.kernel_func = build_act(kernel_func, inplace=False)# False

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())

        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        score = self.embed(x).reshape(B,-1) # 32 * 32
        value,index = torch.topk(score,int(score.shape[-1] ** 0.5))
        zero = torch.zeros(score.shape).to(index.device)
        # zero[index] = 1.
        flat_index = index + torch.arange(0, score.shape[-1] * B, score.shape[-1] ).unsqueeze(1).to(index.device)
        zero.view(-1)[flat_index.view(-1)] = 1.
        zero = F.interpolate(zero.reshape(B,1,int(score.shape[-1] ** 0.5),int(score.shape[-1] ** 0.5)),size=[H,W])

        multi_scale_qkv = torch.reshape(
            multi_scale_qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )

        res_v = multi_scale_qkv.transpose(-1,-2)[..., 2 * self.dim:].clone()

        shape = multi_scale_qkv.shape

        zero = zero.reshape(B,1,H*W).unsqueeze(1)
        zero_x = zero.repeat([1,multi_scale_qkv.shape[1],res_v.shape[-1],1])
        zero = zero.repeat([1,multi_scale_qkv.shape[1],multi_scale_qkv.shape[2],1])
        multi_scale_qkv = multi_scale_qkv[zero > 0].reshape(B,shape[1],3*self.dim,-1)


        multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
        q, k, v = (
            multi_scale_qkv[..., 0: self.dim].clone(),
            multi_scale_qkv[..., self.dim: 2 * self.dim].clone(),
            multi_scale_qkv[..., 2 * self.dim:].clone(),
        )

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

        # x = x.reshape(B,-1,H*W).unsqueeze(1).repeat([1,out.shape[1],1,1])

        # final projecttion
        out = torch.transpose(out, -1, -2)
        res_v = res_v.transpose(-1,-2)
        res_v[zero_x>0] += out.reshape(-1)
        out = torch.reshape(res_v, (B, -1, H, W))
        out = self.proj(out)
        return out

class LiteGatedMSA(nn.Module):
    r""" Lightweight multi-scale attention """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: Optional[int] = None,
            heads_ratio: float = 1.0,
            dim=8,
            use_bias=False,
            norm=(None, "bn2d"),
            act_func=(None, None),
            kernel_func="relu",
            scales: Tuple[int, ...] = (5,),
    ):
        super(LiteGatedMSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim, 3 * total_dim, scale, padding=get_same_padding(scale), groups=3 * total_dim,
                        bias=use_bias[0],
                        ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),

                    nn.Conv2d(
                        3 * total_dim, 3 * total_dim, 1, groups=3 * total_dim,
                        bias=use_bias[0],),
                )
                for scale in scales
            ]
        )



        self.kernel_func = build_act(kernel_func, inplace=False)# False

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())

        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        multi_scale_qkv = torch.reshape(
            multi_scale_qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
        q, k, v = (
            multi_scale_qkv[..., 0: self.dim].clone(),
            multi_scale_qkv[..., self.dim: 2 * self.dim].clone(),
            multi_scale_qkv[..., 2 * self.dim:].clone(),
        )

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

        # final projecttion
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        out = self.proj(out)

        return out

class LiteGatedSA(nn.Module):
    r""" 多head用来同时计算一个attention防止矩阵秩退化 """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: Optional[int] = None,
            heads_ratio: float = 1.0,
            dim=8,
            use_bias=False,
            norm=(None, "bn2d"),
            act_func=(None, None),
            kernel_func="relu",
            scales: Tuple[int, ...] = (5,),
    ):
        super(LiteGatedSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.pre_norm = build_norm("ln",num_features=in_channels)

        self.dim = dim
        self.qkvu = ConvLayer(
            in_channels,
            4 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        4 * total_dim, 4 * total_dim, scale, padding=get_same_padding(scale), groups=4 * total_dim,
                        bias=use_bias[0],
                        ),
                    nn.Conv2d(4 * total_dim, 4 * total_dim, 1, groups=4*heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            )
        self.total_dim = total_dim

    def reverse_channel_shuffle(self,x):
        B,C,H,W = x.shape
        inner_channel = self.total_dim
        groups = C // inner_channel
        x = x.view(B,inner_channel,groups,H,W).transpose(1,2).contiguous().view(B,C,H,W)
        return x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())

        # if self.pre_norm:
        #     x=x.permute(0,2,3,1).reshape(B,H*W,-1)
        #     x = self.pre_norm(x)
        #     x=x.reshape(B,H,W,-1).permute(0,3,1,2)
        # generate multi-scale q, k, v

        qkvu = self.qkvu(x)
        multi_scale_qkvu = [qkvu]
        for op in self.aggreg:
            multi_scale_qkvu.append(op(qkvu))
        multi_scale_qkvu = torch.cat(multi_scale_qkvu, dim=1)
        # reversed shuffle (shuffle net reversed version)
        # multi_scale_qkv = self.reverse_channel_shuffle(multi_scale_qkv) #8 384 32 32

        multi_scale_qkvu = torch.reshape(
            multi_scale_qkvu,
            (
                B,
                -1,
                4 * self.total_dim,
                H * W,
            ),
        )
        multi_scale_qkvu = torch.transpose(multi_scale_qkvu, -1, -2)
        q, k, v, u = (
            multi_scale_qkvu[..., 0: self.total_dim].clone(),
            multi_scale_qkvu[..., self.total_dim: 2 * self.total_dim].clone(),
            multi_scale_qkvu[..., 2 * self.total_dim:3 * self.total_dim].clone(),
            multi_scale_qkvu[..., 3 * self.total_dim:].clone(),
        )

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=0)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        # out = out[..., :-1] / (out[..., -1:] + 1e-15)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

        # final projecttion
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        u = torch.reshape(torch.transpose(u, -1, -2), (B, -1, H, W))
        out = self.proj(out*u)

        return out

class LiteGatedWindowMSA(nn.Module):
    r""" Lightweight multi-scale attention """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: Optional[int] = None,
            heads_ratio: float = 1.0,
            dim=8,
            use_bias=False,
            norm=(None, "bn2d"),
            act_func=(None, None),
            kernel_func="relu",
            scales: Tuple[int, ...] = (3,5,),
    ):
        super(LiteGatedWindowMSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    # nn.Conv2d(
                    #     3 * total_dim, 3 * total_dim, scale, padding=get_same_padding(scale), groups=3 * total_dim,
                    #     bias=use_bias[0],
                    #     ),
                    nn.AvgPool2d(kernel_size=scale,stride=1,padding=get_same_padding(scale),),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),

                    # nn.Conv2d(
                    #     3 * total_dim, 3 * total_dim, 1, groups=3 * total_dim,
                    #     bias=use_bias[0],),
                )
                for scale in scales
            ]
        )



        # self.kernel_func = build_act(kernel_func, inplace=False)# False
        self.kernel_func = nn.Sequential(build_act(kernel_func, inplace=False),
                                         # nn.MaxPool2d(kernel_size=3,stride=1,padding=get_same_padding(3))
                                         )

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            )
        self.window_list = [64,32,32,32]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = list(x.size())
        window_idx = C//16-1
        h_w = self.window_list[window_idx]
        w_w = self.window_list[window_idx]

        x = img2windowsCHW(x, h_w, w_w)
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        multi_scale_qkv = torch.reshape(
            multi_scale_qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
        q, k, v = (
            multi_scale_qkv[..., 0: self.dim].clone(),
            multi_scale_qkv[..., self.dim: 2 * self.dim].clone(),
            multi_scale_qkv[..., 2 * self.dim:].clone(),
        )

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=0)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

        # final projecttion
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, h_w, w_w))
        out = windows2imgCHW(out, B, h_w, w_w, H, W)
        out = self.proj(out)

        return out

class LiteGatedWindowMSA2(nn.Module):
    r""" Lightweight multi-scale attention """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: Optional[int] = None,
            heads_ratio: float = 1.0,
            dim=8,
            use_bias=False,
            norm=(None, "bn2d"),
            act_func=(None, None),
            kernel_func="relu",
            scales: Tuple[int, ...] = (5,),
    ):
        super(LiteGatedWindowMSA2, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            4 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        4 * total_dim, 4 * total_dim, scale, padding=get_same_padding(scale), groups=4 * total_dim,
                        bias=use_bias[0],
                        ),
                    nn.Conv2d(4 * total_dim, 4 * total_dim, 1, groups=4 * heads, bias=use_bias[0]),

                    nn.Conv2d(
                        4 * total_dim, 4 * total_dim, 1, groups=4 * total_dim,
                        bias=use_bias[0],),
                )
                for scale in scales
            ]
        )



        self.kernel_func = build_act(kernel_func, inplace=False)# False

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())
        x = img2windowsCHW(x, H//4, W//4)
        # generate multi-scale q, k, v
        qkvu = self.qkv(x)
        multi_scale_qkv = [qkvu]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkvu))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        multi_scale_qkv = torch.reshape(
            multi_scale_qkv,
            (
                B,
                -1,
                4 * self.dim,
                H * W,
            ),
        )
        multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
        q, k, v ,u = (
            multi_scale_qkv[..., 0: self.dim].clone(),
            multi_scale_qkv[..., self.dim: 2 * self.dim].clone(),
            multi_scale_qkv[..., 2 * self.dim:3 * self.dim].clone(),
            multi_scale_qkv[..., 3 * self.dim:].clone(),
        )

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

        # final projecttion
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H//4, W//4))
        out = windows2imgCHW(out, B, H//4, W//4, H, W)
        u = windows2imgCHW(u, B, H//4, W//4, H, W)
        out = self.proj(out*u)

        return out

class LiteGatedWindowSA(nn.Module):
    r""" 多head用来同时计算一个attention防止矩阵秩退化 """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: Optional[int] = None,
            heads_ratio: float = 1.0,
            dim=8,
            use_bias=False,
            norm=(None, "bn2d"),
            act_func=(None, None),
            kernel_func="relu",
            scales: Tuple[int, ...] = (5,),
    ):
        super(LiteGatedWindowSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.pre_norm = build_norm("ln",num_features=in_channels)

        self.dim = dim
        self.qkvu = ConvLayer(
            in_channels,
            4 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        4 * total_dim, 4 * total_dim, scale, padding=get_same_padding(scale), groups=4 * total_dim,
                        bias=use_bias[0],
                        ),
                    nn.Conv2d(4 * total_dim, 4 * total_dim, 1, groups=4*heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            )
        self.total_dim = total_dim

    def reverse_channel_shuffle(self,x):
        B,C,H,W = x.shape
        inner_channel = self.total_dim
        groups = C // inner_channel
        x = x.view(B,inner_channel,groups,H,W).transpose(1,2).contiguous().view(B,C,H,W)
        return x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())
        x = img2windows(x, H//4, W//4)


        # if self.pre_norm:
        #     x=x.permute(0,2,3,1).reshape(B,H*W,-1)
        #     x = self.pre_norm(x)
        #     x=x.reshape(B,H,W,-1).permute(0,3,1,2)
        # generate multi-scale q, k, v

        qkvu = self.qkvu(x)
        multi_scale_qkvu = [qkvu]
        for op in self.aggreg:
            multi_scale_qkvu.append(op(qkvu))
        multi_scale_qkvu = torch.cat(multi_scale_qkvu, dim=1)
        # reversed shuffle (shuffle net reversed version)
        # multi_scale_qkv = self.reverse_channel_shuffle(multi_scale_qkv) #8 384 32 32

        multi_scale_qkvu = torch.reshape(
            multi_scale_qkvu,
            (
                B,
                -1,
                4 * self.total_dim,
                H * W,
            ),
        )
        multi_scale_qkvu = torch.transpose(multi_scale_qkvu, -1, -2)
        q, k, v, u = (
            multi_scale_qkvu[..., 0: self.total_dim].clone(),
            multi_scale_qkvu[..., self.total_dim: 2 * self.total_dim].clone(),
            multi_scale_qkvu[..., 2 * self.total_dim:3 * self.total_dim].clone(),
            multi_scale_qkvu[..., 3 * self.total_dim:].clone(),
        )

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=0)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        # out = out[..., :-1] / (out[..., -1:] + 1e-15)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

        # final projecttion
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        u = torch.reshape(torch.transpose(u, -1, -2), (B, -1, H, W))
        out = self.proj(out*u)
        out = windows2img(out, H//4, W//4, H, W)


        return out



class LiteFocusMSA(nn.Module):
    r""" Lightweight multi-scale attention """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: Optional[int] = None,
            heads_ratio: float = 1.0,
            dim=8,
            use_bias=False,
            norm=(None, "bn2d"),
            act_func=(None, None),
            kernel_func="relu",
            scales: Tuple[int, ...] = (5,),
            focusing_factor=3,
    ):
        super(LiteFocusMSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        self.heads = heads
        self.split_size = 8
        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim, 3 * total_dim, scale, padding=get_same_padding(scale), groups=3 * total_dim,
                        bias=use_bias[0],
                        ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )

        self.get_v = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, self.dim)))
        self.focusing_factor = focusing_factor
        self.kernel_func = build_act(kernel_func, inplace=False)# False
        self.dwc = nn.Conv2d(in_channels=heads, out_channels=heads, kernel_size=5,
                             groups=heads, padding=5 // 2)
        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            )
    def get_resolusion(self,H,W,idx=-1):
        if idx == -1:
            H_sp, W_sp = H, H
        elif idx == 0:
            H_sp, W_sp = H, self.split_size
        elif idx == 1:
            W_sp, H_sp = W, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        return H_sp,W_sp

    def im2cswin(self, x,H_sp,W_sp):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, H_sp, W_sp)
        # x = x.reshape(-1, self.H_sp * self.W_sp, C).contiguous()
        return x

    def get_lepe(self, x, func,H_sp,W_sp):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        # H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, C // self.num_heads, H_sp * W_sp).permute(0, 2, 1).contiguous()

        x = x.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()
        return x, lepe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())

        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        multi_scale_qkv = torch.reshape(
            multi_scale_qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
        q, k, v = (
            multi_scale_qkv[..., 0: self.dim].clone(),
            multi_scale_qkv[..., self.dim: 2 * self.dim].clone(),
            multi_scale_qkv[..., 2 * self.dim:].clone(),
        )
        Bm,Sm,Lm,Cm = q.shape
        q,k,v = q.reshape(Bm*Sm,Lm,Cm),k.reshape(Bm*Sm,Lm,Cm),v.reshape(Bm*Sm,Lm,Cm)
        H_sp,W_sp = self.get_resolusion(H,W)
        q = self.im2cswin(q,H_sp,W_sp)
        k = self.im2cswin(k,H_sp,W_sp)
        v = self.im2cswin(v,H_sp,W_sp)
        # v = F.pad(v, (0, 1), mode="constant", value=1)
        # v, lepe = self.get_lepe(v, self.get_v,H_sp,W_sp)

        # k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        scale = nn.Softplus()(self.scale)

        # lightweight global attention
        q = self.kernel_func(q) +1e-6
        k = self.kernel_func(k) +1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        feature_map = rearrange(v, "b (h w) c -> b c h w", h=H_sp, w=W_sp)
        feature_map = rearrange(self.dwc(feature_map), "b c h w -> b (h w) c")
        x = x + feature_map
        # x = x + lepe
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.heads)
        out = windows2img(x, H_sp, W_sp, H, W).view(B, -1, self.dim)
        out = out.reshape(Bm,Sm,Lm,Cm).permute(0,1,3,2).reshape(B,-1,H,W)

        out = self.proj(out)

        return out
class FocusedLinearAttention(nn.Module):
    def __init__(self, in_channels, resolution, idx, split_size=7, out_channels=None, heads=4, attn_drop=0., proj_drop=0.,
                 qk_scale=None, focusing_factor=3, kernel_size=5):
        super().__init__()
        self.dim = in_channels
        self.dim_out = out_channels or in_channels
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = heads
        head_dim = in_channels // heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5
        dim = 4
        total_dim = heads * dim
        norm=(None, "bn2d"),
        act_func=(None, None),
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=False,
            norm=norm[0],
            act_func=act_func[0],
            )
        stride = 1
        self.get_v = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)

        self.attn_drop = nn.Dropout(attn_drop)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, in_channels)))
        # self.positional_encoding = nn.Parameter(torch.zeros(size=(1, self.H_sp * self.W_sp, in_channels)))
        # print('Linear Attention {}x{} f{} kernel{}'.
        #       format(H_sp, W_sp, focusing_factor, kernel_size))

    def get_resolusion(self,x,idx=0):
        B,C,H,W = x.shape
        if idx == -1:
            H_sp, W_sp = H, H
        elif idx == 0:
            H_sp, W_sp = H, self.split_size
        elif idx == 1:
            W_sp, H_sp = W, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        return H_sp,W_sp

    def im2cswin(self, x,H_sp,W_sp):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, H_sp, W_sp)
        # x = x.reshape(-1, self.H_sp * self.W_sp, C).contiguous()
        return x

    def get_lepe(self, x, func,H_sp,W_sp):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        # H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, C // self.num_heads, H_sp * W_sp).permute(0, 2, 1).contiguous()

        x = x.reshape(-1, C, H_sp * W_sp).permute(0, 2, 1).contiguous()
        return x, lepe

    def forward(self, x):
        """
        x: B L C
        """
        qkv = self.qkv(x)
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q,self.get_resolusion(q))
        k = self.im2cswin(k,self.get_resolusion(k))
        v, lepe = self.get_lepe(v, self.get_v,self.get_resolusion(v))
        # q, k, v = (rearrange(x, "b h n c -> b n (h c)", h=self.num_heads) for x in [q, k, v])

        # k = k + self.positional_encoding

        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        feature_map = rearrange(v, "b (h w) c -> b c h w", h=self.H_sp, w=self.W_sp)
        feature_map = rearrange(self.dwc(feature_map), "b c h w -> b (h w) c")
        x = x + feature_map
        x = x + lepe
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)

        return x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm

def img2windowsCHW(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

def windows2imgCHW(img_splits_hw,B, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' C H W
    """

    img = img_splits_hw.reshape(B, -1, H // H_sp, W // W_sp, H_sp, W_sp)
    img = img.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    return img

class Lite4xMSA(nn.Module):
    r""" 多head用来同时计算一个attention防止矩阵秩退化 """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: Optional[int] = None,
            heads_ratio: float = 1.0,
            dim=8,
            use_bias=False,
            norm=(None, "bn2d"),
            act_func=(None, None),
            kernel_func="relu",
            scales: Tuple[int, ...] = (5,),
    ):
        super(Lite4xMSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.pre_norm = build_norm("ln",num_features=in_channels)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim, 3 * total_dim, scale, padding=get_same_padding(scale), groups=3 * total_dim,
                        bias=use_bias[0],
                        ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3*heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            )
        self.total_dim = total_dim

    def reverse_channel_shuffle(self,x):
        B,C,H,W = x.shape
        inner_channel = self.total_dim
        groups = C // inner_channel
        x = x.view(B,inner_channel,groups,H,W).transpose(1,2).contiguous().view(B,C,H,W)
        return x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())
        # if self.pre_norm:
        #     x=x.permute(0,2,3,1).reshape(B,H*W,-1)
        #     x = self.pre_norm(x)
        #     x=x.reshape(B,H,W,-1).permute(0,3,1,2)
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)
        # reversed shuffle (shuffle net reversed version)
        # multi_scale_qkv = self.reverse_channel_shuffle(multi_scale_qkv) #8 384 32 32

        multi_scale_qkv = torch.reshape(
            multi_scale_qkv,
            (
                B,
                -1,
                3 * self.total_dim,
                H * W,
            ),
        )
        multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
        q, k, v = (
            multi_scale_qkv[..., 0: self.total_dim].clone(),
            multi_scale_qkv[..., self.total_dim: 2 * self.total_dim].clone(),
            multi_scale_qkv[..., 2 * self.total_dim:].clone(),
        )

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode="constant", value=0)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        # out = out[..., :-1] / (out[..., -1:] + 1e-15)
        out = out[..., :-1] / (out[..., -1:] + 1e-15)

        # final projecttion
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        out = self.proj(out)

        return out

class LiteFlowMSA(nn.Module):
    r""" Lightweight multi-scale attention """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: Optional[int] = None,
            heads_ratio: float = 1.0,
            dim=8,
            use_bias=False,
            norm=(None, "bn2d"),
            act_func=(None, None),
            kernel_func="sigmoid",
            scales: Tuple[int, ...] = (5,),
    ):
        super(LiteFlowMSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim, 3 * total_dim, scale, padding=get_same_padding(scale), groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

        self.num_heads = heads
        self.softmax = nn.Softmax(dim=-1)

    def my_sum(self, a, b):
        # "nhld,nhd->nhl"
        return torch.sum(a * b[:, :, None, :], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())

        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        multi_scale_qkv = torch.reshape(
            multi_scale_qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
        q, k, v = (
            multi_scale_qkv[..., 0: self.dim].clone(),
            multi_scale_qkv[..., self.dim: 2 * self.dim].clone(),
            multi_scale_qkv[..., 2 * self.dim:].clone(),
        )

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        sink_incoming = 1.0 / (self.my_sum(q + 1e-6, k.sum(dim=2) + 1e-6) + 1e-6)
        source_outgoing = 1.0 / (self.my_sum(k + 1e-6, q.sum(dim=2) + 1e-6) + 1e-6)
        conserved_sink = self.my_sum(q + 1e-6, (k * source_outgoing[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = self.my_sum(k + 1e-6, (q * sink_incoming[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(q.shape[2]) / float(k.shape[2])))
        # competition
        source_competition = torch.softmax(conserved_source, dim=-1) * float(k.shape[2])

        trans_k = k.transpose(-1, -2)

        #v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, (v * source_competition[:, :, :, None]))
        out = (torch.matmul(q, kv)* sink_incoming[:, :, :, None]) * sink_allocation[:, :, :, None]
        #out = out[..., :-1] / (out[..., -1:] + 1e-15)

        if q.shape[2] == 1024:
            with torch.no_grad():
                vis = ((q@trans_k).transpose(-1, -2).reshape(B, -1, q.shape[2], q.shape[2]))[:,0,:,:].unsqueeze(1)
                vutils.save_image(vis, f"output/attan.png")

        # final projecttion
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        out = self.proj(out)

        return out

    class LiteFlowMSA(nn.Module):
        r""" Lightweight multi-scale attention """

        def __init__(
                self,
                in_channels: int,
                out_channels: int,
                heads: Optional[int] = None,
                heads_ratio: float = 1.0,
                dim=8,
                use_bias=False,
                norm=(None, "bn2d"),
                act_func=(None, None),
                kernel_func="sigmoid",
                scales: Tuple[int, ...] = (5,),
        ):
            super(LiteFlowMSA, self).__init__()
            heads = heads or int(in_channels // dim * heads_ratio)

            total_dim = heads * dim

            use_bias = val2tuple(use_bias, 2)
            norm = val2tuple(norm, 2)
            act_func = val2tuple(act_func, 2)

            self.dim = dim
            self.qkv = ConvLayer(
                in_channels,
                3 * total_dim,
                1,
                use_bias=use_bias[0],
                norm=norm[0],
                act_func=act_func[0],
            )
            self.aggreg = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            3 * total_dim, 3 * total_dim, scale, padding=get_same_padding(scale), groups=3 * total_dim,
                            bias=use_bias[0],
                        ),
                        nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                    )
                    for scale in scales
                ]
            )
            self.kernel_func = build_act(kernel_func, inplace=False)

            self.out_norm = nn.LayerNorm(dim)

            self.proj = ConvLayer(
                total_dim * (1 + len(scales)),
                out_channels,
                1,
                use_bias=use_bias[1],
                norm=norm[1],
                act_func=act_func[1],
            )

            self.num_heads = heads
            self.softmax = nn.Softmax(dim=-1)

        def my_sum(self, a, b):
            # "nhld,nhd->nhl"
            return torch.sum(a * b[:, :, None, :], dim=-1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, _, H, W = list(x.size())

            # generate multi-scale q, k, v
            qkv = self.qkv(x)
            multi_scale_qkv = [qkv]
            for op in self.aggreg:
                multi_scale_qkv.append(op(qkv))
            multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

            multi_scale_qkv = torch.reshape(
                multi_scale_qkv,
                (
                    B,
                    -1,
                    3 * self.dim,
                    H * W,
                ),
            )
            multi_scale_qkv = torch.transpose(multi_scale_qkv, -1, -2)
            q, k, v = (
                multi_scale_qkv[..., 0: self.dim].clone(),
                multi_scale_qkv[..., self.dim: 2 * self.dim].clone(),
                multi_scale_qkv[..., 2 * self.dim:].clone(),
            )

            # lightweight global attention
            q = self.kernel_func(q)
            k = self.kernel_func(k)
            sink_incoming = 1.0 / (self.my_sum(q + 1e-6, k.sum(dim=2) + 1e-6) + 1e-6)
            source_outgoing = 1.0 / (self.my_sum(k + 1e-6, q.sum(dim=2) + 1e-6) + 1e-6)
            conserved_sink = self.my_sum(q + 1e-6, (k * source_outgoing[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
            conserved_source = self.my_sum(k + 1e-6, (q * sink_incoming[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
            conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
            # allocation
            sink_allocation = torch.sigmoid(conserved_sink * (float(q.shape[2]) / float(k.shape[2])))
            # competition
            source_competition = torch.softmax(conserved_source, dim=-1) * float(k.shape[2])

            trans_k = k.transpose(-1, -2)

            # v = F.pad(v, (0, 1), mode="constant", value=1)
            kv = torch.matmul(trans_k, (v * source_competition[:, :, :, None]))
            out = (torch.matmul(q, kv) * sink_incoming[:, :, :, None]) * sink_allocation[:, :, :, None]
            # out = out[..., :-1] / (out[..., -1:] + 1e-15)
            out = self.out_norm(out)
            # final projecttion
            out = torch.transpose(out, -1, -2)
            out = torch.reshape(out, (B, -1, H, W))
            out = self.proj(out)

            return out

class LiteCrossFlowMSA(nn.Module):
    r""" Lightweight multi-scale attention """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            heads: Optional[int] = None,
            heads_ratio: float = 1.0,
            dim=8,
            use_bias=False,
            norm=(None, "bn2d"),
            act_func=(None, None),
            kernel_func="sigmoid",
            scales: Tuple[int, ...] = (5,),
    ):
        super(LiteCrossFlowMSA, self).__init__()
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.q = ConvLayer(
            in_channels,
            total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.kv = ConvLayer(
            in_channels,
            2 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg_q = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        total_dim, total_dim, scale, padding=get_same_padding(scale), groups=total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(total_dim, total_dim, 1, groups=heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )

        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        2 * total_dim, 2 * total_dim, scale, padding=get_same_padding(scale), groups=2 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(2 * total_dim, 2 * total_dim, 1, groups=2 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.out_norm = nn.LayerNorm(dim)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

        self.num_heads = heads
        self.softmax = nn.Softmax(dim=-1)

    def my_sum(self, a, b):
        # "nhld,nhd->nhl"
        return torch.sum(a * b[:, :, None, :], dim=-1)

    def forward(self, x: torch.Tensor,edge: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(x.size())

        # generate multi-scale q, k, v
        q = self.q(edge)
        multi_scale_q = [q]
        for op in self.aggreg_q:
            multi_scale_q.append(op(q))
        multi_scale_q = torch.cat(multi_scale_q, dim=1)
        multi_scale_q = torch.reshape(
            multi_scale_q,
            (
                B,
                -1,
                self.dim,
                H * W,
            ),
        )

        kv = self.kv(x)
        multi_scale_kv = [kv]
        for op in self.aggreg:
            multi_scale_kv.append(op(kv))
        multi_scale_kv = torch.cat(multi_scale_kv, dim=1)
        multi_scale_kv = torch.reshape(
            multi_scale_kv,
            (
                B,
                -1,
                2 * self.dim,
                H * W,
            ),
        )
        multi_scale_q = torch.transpose(multi_scale_q, -1, -2)
        multi_scale_kv = torch.transpose(multi_scale_kv, -1, -2)
        q = multi_scale_q
        k, v = (
            multi_scale_kv[..., 0: self.dim].clone(),
            multi_scale_kv[..., self.dim: 2 * self.dim].clone(),
        )

        # lightweight global attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        sink_incoming = 1.0 / (self.my_sum(q + 1e-6, k.sum(dim=2) + 1e-6) + 1e-6)
        source_outgoing = 1.0 / (self.my_sum(k + 1e-6, q.sum(dim=2) + 1e-6) + 1e-6)
        conserved_sink = self.my_sum(q + 1e-6, (k * source_outgoing[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = self.my_sum(k + 1e-6, (q * sink_incoming[:, :, :, None]).sum(dim=2) + 1e-6) + 1e-6
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
        # allocation
        sink_allocation = torch.sigmoid(conserved_sink * (float(q.shape[2]) / float(k.shape[2])))
        # competition
        source_competition = torch.softmax(conserved_source, dim=-1) * float(k.shape[2])

        trans_k = k.transpose(-1, -2)

        #v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, (v * source_competition[:, :, :, None]))
        out = (torch.matmul(q, kv)* sink_incoming[:, :, :, None]) * sink_allocation[:, :, :, None]
        #out = out[..., :-1] / (out[..., -1:] + 1e-15)
        #norm devil in transformer
        out = self.out_norm(out)
        # final projecttion
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        out = self.proj(out)

        return out

class EfficientViTBlock(nn.Module):
    def __init__(self, in_channels: int, heads_ratio: float = 1.0, dim=32, expand_ratio: float = 4, norm="bn2d",
                 act_func="hswish"):#hswish
        super(EfficientViTBlock, self).__init__()
        self.context_module = ResidualBlock(
            LiteMSA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                heads=4,
                norm=(None, norm),
            ),
            IdentityLayer(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(False, False, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x

class EfficientVCTBlock(nn.Module):
    def __init__(self, in_channels: int, heads_ratio: float = 1.0, dim=32, expand_ratio: float = 4, norm="bn2d",
                 act_func="hswish"):#hswish
        super(EfficientVCTBlock, self).__init__()
        self.context_module =LiteMCA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                heads=4,
                norm=(None, norm),
            )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(False, False, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor,y: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x,y)
        x = self.local_module(x)
        return x

class EfficientGatedViTBlock(nn.Module):
    def __init__(self, in_channels: int, heads_ratio: float = 1.0, dim=32, expand_ratio: float = 4, norm="bn2d",
                 act_func="hswish"):#hswish
        super(EfficientGatedViTBlock, self).__init__()
        self.context_module = ResidualBlock(
            LiteTopkMSA( #LiteGatedMSA
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                heads=4,
                norm=(norm, norm),
            ),
            IdentityLayer(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(False, False, False),
            norm=(norm, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x

class EfficientGatedWindowViTBlock(nn.Module):
    def __init__(self, in_channels: int, heads_ratio: float = 1.0, dim=32, expand_ratio: float = 4, norm="bn2d",
                 act_func="relu"):#hswish
        super(EfficientGatedWindowViTBlock, self).__init__()
        self.context_module = ResidualBlock(
            LiteGatedWindowMSA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                heads=4,
                norm=(None, norm),
            ),
            IdentityLayer(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(False, False, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class EfficientCrossViTBlock(nn.Module):
    def __init__(self, in_channels: int, heads_ratio: float = 1.0, dim=32, expand_ratio: float = 4, norm="bn2d",
                 act_func="hswish"):
        super(EfficientCrossViTBlock, self).__init__()
        self.context_module = ResidualCrossBlock(
            LiteCrossFlowMSA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                # heads=3,
                dim=dim,
                norm=(None, norm),
            ),
            IdentityLayer(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor,edge:torch.Tensor) -> torch.Tensor:
        x = self.context_module(x,edge)
        x = self.local_module(x)
        return x

#################################################################################
#                             Functional Blocks                                 #
#################################################################################


class ResidualBlock(nn.Module):
    def __init__(
            self,
            main: Optional[nn.Module],
            shortcut: Optional[nn.Module],
            post_act=None,
            pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res

class ResidualCrossBlock(nn.Module):
    def __init__(
            self,
            main: Optional[nn.Module],
            shortcut: Optional[nn.Module],
            post_act=None,
            pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualCrossBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor,edge: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x,edge)
        else:
            return self.main(self.pre_norm(x),self.pre_norm(edge))

    def forward(self, x: torch.Tensor,edge: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x,edge)
        else:
            res = self.forward_main(x,edge) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class DAGBlock(nn.Module):
    def __init__(
            self,
            inputs: Dict[str, nn.Module],
            merge_mode: str,
            post_input: Optional[nn.Module],
            middle: nn.Module,
            outputs: Dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge_mode = merge_mode
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        feat = merge_tensor(feat, self.merge_mode, dim=1)
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class OpSequential(nn.Module):
    def __init__(self, op_list: List[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


class EfficientViTBackbone(nn.Module):
    def __init__(self, width_list: List[int], depth_list: List[int], in_channels=1, dim=32, expand_ratio=4, norm="bn2d",
                 act_func="hswish") -> None:
        super().__init__()

        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    EfficientViTBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)
        #self.channel = [i.size(1) for i in self.forward(torch.randn(8, 1, 512, 512))]

    @staticmethod
    def build_local_block(in_channels: int, out_channels: int, stride: int, expand_ratio: float, norm: str,
                          act_func: str, fewer_norm: bool = False) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(False, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(False, False, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        res = []
        x = self.input_stem(x)
        res.append(x)
        for stage_id, stage in enumerate(self.stages, 1):
            x = stage(x)
            res.append(x)
        return res


def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        k = k[9:]
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict


def efficientvit_b0(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']))
    return backbone


def efficientvit_b1(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[16, 32, 64, 128, 256],
        # depth_list=[1, 2, 3, 3, 4],
        depth_list=[1, 2, 2, 2, 1],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']))
    return backbone

def efficientvit_bs(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[16, 32, 64, 128],
        # depth_list=[1, 2, 3, 3, 4],
        depth_list=[1, 1, 1, 2],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']),strict=False)
    return backbone

def efficientvit_bs16(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[16, 16, 16, 16],
        # depth_list=[1, 2, 3, 3, 4],
        depth_list=[1, 1, 1, 1],
        dim=16,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']),strict=False)
    return backbone

def efficientvit_bs_32(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[32,64,128,256],
        # depth_list=[1, 2, 3, 3, 4],
        depth_list=[3, 3, 3, 3],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']))
    return backbone

def efficientvit_b2(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']))
    return backbone


def efficientvit_b3(weights='', **kwargs) -> EfficientViTBackbone:
    backbone = EfficientViTBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, EfficientViTBackbone),
    )
    if weights:
        backbone.load_state_dict(update_weight(backbone.state_dict(), torch.load(weights)['state_dict']))
    return backbone


if __name__ == '__main__':
    model = efficientvit_b1()
    inputs = torch.randn((1, 1, 512, 512))
    res = model(inputs)
    for i in res:
        print(i.size())