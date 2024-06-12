import torch
import torch.nn as nn
from .LKBackbone_lr_psd import CSPNeXt
from mmdet.models.necks.fpn import FPN
import torch.nn.functional as F
from torchvision.ops import RoIAlign
import numpy as np
import cv2
from scipy import ndimage
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

def get_larger_box(img,x,y,w,h,scale_factor = 1.5):
    max_value = img.shape[-2:]
    cx = (x + x + w) / 2
    cy = (y + y + h) / 2
    new_w = w * 1.5
    new_h = h * 1.5
    new_x1 = cx - new_w / 2
    new_y1 = cy - new_h / 2
    new_x2 = cx + new_w / 2
    new_y2 = cy + new_h / 2
    new_x1 = max(0, min(new_x1, max_value[-1]))
    new_y1 = max(0, min(new_y1, max_value[-2]))
    new_x2 = max(0, min(new_x2, max_value[-1]))
    new_y2 = max(0, min(new_y2, max_value[-2]))

    return [new_x1,new_y1,new_x2,new_y2]

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
            nn.Conv2d(self.backbone_channels[-4],self.backbone_channels[-4],3,1,1),
        ],
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(
                self.backbone_channels[-4],
                8,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(8),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.sparse_head.append(self.upsample)

        self.last = nn.Sequential(
            nn.Conv2d(
                8,
                4,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.last2 = nn.Sequential(

            nn.Conv2d(
                4,
                1,
                kernel_size=5,
                stride=1,
                padding=5//2
            ),
        )


        self.up2x = nn.Upsample(scale_factor=2,mode='bilinear')
        self.up8x = nn.Upsample(scale_factor=1/2,mode='nearest')
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
        self.roi_align = RoIAlign(output_size=16, spatial_scale=0.5, sampling_ratio=-1)
        self.roi_align1 = RoIAlign(output_size=16, spatial_scale=1, sampling_ratio=-1)

        self.roi_num = self.backbone_channels[-4]
        self.feats_branch = nn.Sequential(
            nn.Conv2d(self.roi_num, self.roi_num, kernel_size=7, stride=1, padding=3), nn.BatchNorm2d(self.roi_num),
            nn.ReLU(),
            nn.Conv2d(self.roi_num, self.roi_num, 3, 1, 1), nn.BatchNorm2d(self.roi_num), nn.ReLU(),
            # ChannelAttention1(64, 64),
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.Conv2d(self.roi_num, self.roi_num, kernel_size=7, stride=1, padding=3), nn.BatchNorm2d(self.roi_num),
            nn.ReLU(),
            nn.Conv2d(self.roi_num, self.roi_num, 3, 1, 1), nn.BatchNorm2d(self.roi_num), nn.ReLU(), )
        self.roi_branch_v1 = nn.Sequential(
            nn.Conv2d(self.roi_num + 1, self.roi_num, kernel_size=7, stride=1, padding=3),nn.BatchNorm2d(self.roi_num), nn.ReLU(),
                                           nn.Conv2d(self.roi_num,self.roi_num, 3, 1, 1), nn.BatchNorm2d(self.roi_num), nn.ReLU(),
                                           # ChannelAttention1(64, 64),
                                           # nn.Upsample(scale_factor=4,mode='bilinear'),
                                           nn.Conv2d(self.roi_num, self.roi_num, 3, 1, 1), nn.BatchNorm2d(self.roi_num), nn.ReLU(),
            nn.Conv2d(self.roi_num, 1, 1), nn.Sigmoid(), )
        # self.idx_iter=-1





    def forward(self, input):
        with torch.no_grad():
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
            deep_feat = out

            for m in self.sparse_head:
                out = m(out)

                # out = torch.concat([out, self.shut2x(feats2x)], dim=1)



            out = self.last2(self.last(out))
        once = out


        # TWICE
        b = once.shape[0]
        index = []
        batch_boxes = []  # [x1, y1, x2, y2]
        for i in range(b):
            once_binary = np.array((once[i].squeeze() > 0.5).cpu())
            labeled_instances, num_instances = ndimage.label(once_binary)
            if i == 0:
                num_left = 0
            else:
                num_left = index[i - 1]
            if num_instances == 0:
                index.append(num_left + 1)  # [0,0,0,0]也算一个
            else:
                index.append(num_left + num_instances)
            bounding_boxes = []
            if num_instances > 0:
                for instance_label in range(1, num_instances + 1):  # 从1开始，因为0表示背景
                    instance_region = (labeled_instances == instance_label).astype(np.uint8)
                    # 使用OpenCV的findContours函数找到实例的轮廓
                    contours, _ = cv2.findContours(instance_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # 提取轮廓的边界框
                    if len(contours) > 0:
                        contour = contours[0]
                        x, y, w, h = cv2.boundingRect(contour)
                        bounding_box = get_larger_box(input,x, y, w, h, 1.2)
                        # bounding_box = [x, y, x + w, y + h]
                        bounding_boxes.append(bounding_box)
            else:
                bounding_boxes.append([0, 0, 0, 0])
            batch_boxes.append(torch.tensor(bounding_boxes, dtype=torch.float).cuda())

        x_0_0 = self.feats_branch(F.interpolate(deep_feat, scale_factor=2, mode='bilinear'))

        roi_feature = self.roi_align(x_0_0, batch_boxes)
        roi_once = self.roi_align1(once, batch_boxes)
        roi_feature = torch.concat([roi_once, roi_feature], dim=1)
        roi_feature = self.roi_branch_v1(roi_feature)  # [L(boxes),1,16,16]

        twice = once.clone()
        for i in range(b):
            if i == 0:
                left = 0
            else:
                left = index[i - 1]
            if left == index[i]:
                continue
            boxes_for_this_batch = batch_boxes[i]
            roi_for_this_batch = roi_feature[left:index[i], :, :, :]
            for j in range(index[i] - left):
                x1, y1, x2, y2 = list(boxes_for_this_batch[j].squeeze().to(torch.int64))
                if int(y2 - y1) == 0 or int(x2 - x1) == 0:
                    continue
                twice[i, 0, y1:y2, x1:x2] += F.interpolate(roi_for_this_batch[j].unsqueeze(0),
                                                           size=(int(y2 - y1), int(x2 - x1)), mode='bilinear').reshape(
                    twice[i, 0, y1:y2, x1:x2].shape)
        twice = twice.clamp(0., 1.)

        # return out.sigmoid()

        return [twice, once,  [index, batch_boxes], roi_feature]

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




