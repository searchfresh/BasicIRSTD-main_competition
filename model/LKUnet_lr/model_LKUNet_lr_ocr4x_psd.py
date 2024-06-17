import torch
import torch.nn as nn
# from mmpretrain.models.backbones.mobileone import MobileOne, MobileOneBlock
from .LKBackbone_lr_psd import CSPNeXt
from mmdet.models.necks.cspnext_pafpn import CSPNeXtPAFPN
from mmdet.models.necks.fpn import FPN
from mmpretrain.models.utils.sparse_modules import (SparseAvgPooling, SparseConv2d, SparseHelper,
                                                    SparseMaxPooling)
from mmpretrain.models.backbones.sparse_convnext import SparseConvNeXtBlock
# from mmpretrain import inference_model
import torch.nn.functional as F
from torch.nn.modules.utils import _triple, _pair, _single
import torchvision.utils as vutils
#from visual_mask import visual

class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return nn.BatchNorm2d

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c
        probs = torch.sigmoid(self.scale * probs)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear',)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output



class Competition(nn.Module):
    def __init__(self,in_channels,kernel_size=9):
        super(Competition, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels,in_channels*2,1),
                                   nn.BatchNorm2d(in_channels*2),
                                   nn.ReLU6())
        self.kernel_size = kernel_size
        # self.conv2 = nn.Conv2d(in_channels,1,self.kernel_size,1,padding = self.kernel_size // 2)
        # self.up = nn.PixelShuffle(2)
        self.channs = in_channels

    def forward(self,x):
        x = self.conv1(x)
        x1 = x[:,0:self.channs,:,:]
        x2 = x[:,self.channs:,:,:]
        # x2 = self.conv2(x2).repeat([1,self.channs,1,1]) #resourse

        e_x = torch.exp(x1)
        kernel_size = self.kernel_size
        return F.avg_pool2d(x2.mul(e_x), kernel_size, stride=1, padding = kernel_size // 2).mul_(kernel_size**2).div_(
            F.avg_pool2d(e_x, kernel_size, stride=1, padding = kernel_size // 2).mul_(kernel_size**2))

class Competition2(nn.Module):
    def __init__(self):
        super(Competition2, self).__init__()
        self.down = nn.PixelUnshuffle(downscale_factor=8)
        self.up = nn.PixelShuffle(upscale_factor=8)

    def forward(self,x):
        x = self.down(x) **2
        B,C,H,W = x.shape
        x = x.reshape([B, C, -1]).softmax(dim=-1).reshape(B,C,H,W) * x
        x = self.up(x)
        return x


class LKUNet(nn.Module):
    def __init__(self,):
        super(LKUNet, self).__init__()


        """
            hyper parameter
        """
        self.backbone_channels = [64, 128, 256, 512]
        """"""
        self.backbone_channels_cat = [64, 128, 256, 512]

        self.backbone = CSPNeXt()

        self.neck = FPN(
            in_channels=self.backbone_channels_cat[-3:],
            out_channels=self.backbone_channels[-3],
            # use_depthwise=True,
            act_cfg=dict(type='ReLU'),
            num_outs=len(self.backbone_channels[-3:])
            # use_depthwise=True,
        )

        self.channel_shut1 = nn.Sequential(
            nn.Conv2d(
                in_channels = self.backbone_channels[-3] * 2,
                out_channels = self.backbone_channels[-3],
                kernel_size = 1,
                stride=1
            ),
            nn.BatchNorm2d(self.backbone_channels[-3]),
            nn.ReLU(inplace=True)
        )
        self.channel_shut2 = nn.Sequential(nn.Conv2d(
            in_channels=self.backbone_channels[-3] + self.backbone_channels[-4],
            out_channels= self.backbone_channels[-4],
            kernel_size=1,
            stride=1
        ),
            nn.BatchNorm2d(self.backbone_channels[-4]),
            nn.ReLU(inplace=True)
        )
        self.sparse_head = nn.ModuleList([
            nn.Conv2d(self.backbone_channels[-4],self.backbone_channels[-4],3,1,1),
            nn.BatchNorm2d(self.backbone_channels[-4]),
            nn.ReLU()
        ],
        )

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(self.backbone_channels[-4], self.backbone_channels[-4],
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.backbone_channels[-4]),
            nn.ReLU(inplace=True),
        )

        self.upsamplex = nn.Sequential(
            nn.Conv2d(
                self.backbone_channels[-4],
                self.backbone_channels[-4],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(self.backbone_channels[-4]),
            # nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.sparse_head.append(self.upsamplex)
        self.context_head = nn.Conv2d(self.backbone_channels[-4], 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sparse_head.append(self.context_head)

        self.last = nn.Sequential(
            nn.Conv2d(
                self.backbone_channels[-4],
                16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.last1x = nn.Sequential(
            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.last2x = nn.Sequential(

            nn.Conv2d(
                16,
                1,
                kernel_size=5,
                stride=1,
                padding=5//2
            ),
        )


        self.up2x = nn.Upsample(scale_factor=2,mode='bilinear')
        self.up8x = nn.Upsample(scale_factor=1/2,mode='bilinear')
        self.shut2x = nn.Sequential(
            # # 新增
            # nn.Conv2d(16,out_channels=16,kernel_size=3,stride=1,padding=1),
            # nn.BatchNorm2d(num_features=16),
            # nn.ReLU(inplace=True),

            nn.Conv2d(16,out_channels=8,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True)
        )
        # self.competition = Competition(in_channels=8*2)
        self.competition = Competition2()

        self.ocr_gather_head = SpatialGather_Module(1)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=self.backbone_channels[-4],
                                                 key_channels=self.backbone_channels[-4],
                                                 out_channels=self.backbone_channels[-4],
                                                 scale=1,
                                                 dropout=0.05,
                                                 )

        # self.idx_iter=-1





    def forward(self, input):

        with torch.no_grad():
            layers = self.backbone(input)

            feats4x = layers[0]
            feats = self.neck(layers[-3:])
            feats8x, feats16x, feats32x = feats
            feats8x = self.channel_shut1(torch.concat([feats8x, self.up2x(feats16x)], dim=1))
            feats4x = self.channel_shut2(torch.concat([feats4x, self.up2x(feats8x)], dim=1))
            out = feats4x

            for m in self.sparse_head:
                out = m(out)
            out_aux = out
            feats_ocr = self.conv3x3_ocr(feats4x)
            context = self.ocr_gather_head(feats_ocr, out_aux)
            out = self.ocr_distri_head(feats_ocr, context)

        # out = torch.concat([out, self.shut2x(feats2x)], dim=1)
        # B, C, H, W = feats_ocr.shape
        # out = feats_ocr.reshape([B, C, -1]).softmax(dim=-1).reshape(B, C, H, W) * feats_ocr
        out=self.last(out)
        out = out+self.last1x(out)
        out = self.last2x(out)
        return [out.sigmoid(),F.interpolate(out_aux,scale_factor=2,mode='bilinear').sigmoid()]

    def pixel_shuffle_down(self,x):
        # with torch.no_grad():
        # x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = torch.pixel_unshuffle(x,2)
        return x





from datetime import datetime
if __name__ == '__main__':
    x = torch.randn(8,1,512,512)
    net= RepirDet(deploy=True)

    start_time = datetime.now()
    out = net(x)
    end_time = datetime.now()
    # print(out)
    print("程序执行时间为：", end_time - start_time)




