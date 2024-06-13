import torch
import torch.nn as nn
# from mmpretrain.models.backbones.mobileone import MobileOne, MobileOneBlock
from .LKBackbone_lr_psd_lk import CSPNeXt
from mmdet.models.necks.cspnext_pafpn import CSPNeXtPAFPN
from mmdet.models.necks.fpn import FPN
from mmpretrain.models.utils.sparse_modules import (SparseAvgPooling, SparseConv2d, SparseHelper,
                                                    SparseMaxPooling)
from mmpretrain.models.backbones.sparse_convnext import SparseConvNeXtBlock
# from mmpretrain import inference_model
import torch.nn.functional as F
from torch.nn.modules.utils import _triple, _pair, _single
import torchvision.utils as vutils
# from visual_mask import visual

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
        self.backbone_channels = [64//2, 128//2, 256//2, 512//2]
        """"""
        self.backbone_channels_cat = [64//2 + 0, 128//2 + 64, 256//2 + 64*4, 512//2 + 64*16]

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
            nn.Conv2d(self.backbone_channels[-4],self.backbone_channels[-4],7,1,3),
            nn.BatchNorm2d(num_features=self.backbone_channels[-4]),
            nn.ReLU(inplace=True),
        ],
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(
                self.backbone_channels[-4],
                16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.sparse_head.append(self.upsample)

        self.last = nn.Sequential(
            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.last2 = nn.Sequential(

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
        # self.competition = Competition2()

        # self.idx_iter=-1





    def forward(self, input):
        ps8x = self.pixel_shuffle_down(self.pixel_shuffle_down(self.pixel_shuffle_down(input)))
        ps16x = self.pixel_shuffle_down(ps8x)
        ps32x = self.pixel_shuffle_down(ps16x)

        # start_time = datetime.now()
        layers = self.backbone(input)
        # end_time = datetime.now()
        # print("b：", end_time - start_time)

        feats4x = layers[0]
        # start_time = datetime.now()
        layers[-3] = torch.concat([layers[-3],ps8x], dim=1)
        layers[-2] = torch.concat([layers[-2],ps16x], dim=1)
        layers[-1] = torch.concat([layers[-1],ps32x], dim=1)

        feats = self.neck(layers[-3:])

        feats8x, feats16x, feats32x = feats



        feats8x = self.channel_shut1(torch.concat([feats8x, self.up2x(feats16x)], dim=1))
        feats4x = self.channel_shut2(torch.concat([feats4x, self.up2x(feats8x)], dim=1))
        out = feats4x



        for m in self.sparse_head:
            out = m(out)

        # out = torch.concat([out, self.shut2x(feats2x)], dim=1)


        out = self.last2(self.last(out))
        return out.sigmoid()

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




