import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            # 若列表或者元组，每一个计算loss并且取平均
            loss_total = 0
            for i in range(len(preds)):
                # 新添加修改
                # pred = torch.nn.Sigmoid()(preds[i])
                # pred = torch.sigmoid(preds[i])

                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            # 新添加修改
            # pred = torch.nn.Sigmoid()(preds)
            # pred = torch.sigmoid(preds)
            pred = preds  # 其它方法的head 输出时已经进行过sigmoid
            smooth = 1
            # print(pred.shape)
            # print(gt_masks.shape)
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss

class ISNetLoss(nn.Module):   #为ISNet单独配置的损失
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()
        
    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())
        
        ### img loss
        loss_img = self.softiou(preds[0], gt_masks)
        
        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt)+ self.softiou(preds[1].sigmoid(), edge_gt)
        
        return loss_img + loss_edge



class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            # 若列表或者元组，每一个计算loss并且取平均
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                new = torch.zeros_like(pred)
                mask = pred > 0
                new[mask] = pred[mask]
                pred = new
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum()+smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            # new = torch.zeros_like(pred)
            # mask = pred > 0.5
            # new[mask] = 1.0
            # pred = new
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss
# 新增修改
# class RepirLoss(nn.Module):
#     def __init__(self):
#         super(RepirLoss, self).__init__()
#         self.softiou = SoftIoULoss()
#         # self.softiou = FocalSoftIoULoss()
#         # self.softiou32 = SplitIoULoss(32)
#         self.iou = IoULoss()
#         #self.iou = FocalHardIoULoss1()
#
#         self.bce = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=torch.tensor(10))
#         #self.psloss = pixelsumLoss()
#         self.bce_sum = nn.BCELoss(reduction='sum')
#         # 512*512 --> 2*2
#         self.pool2 = nn.MaxPool2d(kernel_size=256, stride=256)
#         # 512*512 --> 4*4
#         self.pool4 = nn.MaxPool2d(kernel_size=128, stride=128)
#         # 512*512 --> 8*8   256*256 --> 4*4
#         self.pool8 = nn.MaxPool2d(kernel_size=64, stride=64)
#         # 512*512 --> 16*16 256*256 --> 8*8
#         self.pool16 = nn.MaxPool2d(kernel_size=32, stride=32)
#         #                   256*256 --> 16*16
#         self.pool32 = nn.MaxPool2d(kernel_size=16, stride=16)
#         #                   256*256 --> 32*32
#         self.pool64 = nn.MaxPool2d(kernel_size=8, stride=8)
#         self.pool128 = nn.MaxPool2d(kernel_size=4, stride=4)
#         self.pool256 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.layer_lvl_num = 2
#
#         self.threshold = 0.45
#
#         #self.weighting = Aligned_MTL()
#         self.grad = Get_gradient_nopadding()
#         #self.vfocal = VFLoss()
#
#         self.up2x = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
#         self.up4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#         self.up8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
#         self.up16x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
#
#         # self.point_loss = HungarianMatcher(num_points=200)
#         # self.lovaze = LovaszSoftmax()
#         self.focal = nn.BCELoss()
#
#     def forward(self, pred, gt_masks):
#         pred_, classify_result= pred
#         b,_,_,_ = pred_.shape
#
#         with torch.no_grad():
#             if gt_masks.shape[-1] == 512:
#                 classify_gt = self.pool16(gt_masks)
#             else:
#                 classify_gt = self.pool32(gt_masks)
#
#
#         # for 1k
#         # loss_iou = self.iou(pred_, gt_masks)
#         # loss_class = self.iou(classify_result, classify_gt)
#         # loss_total = loss_iou+loss_class
#         # for NUDT
#         loss_total = self.softiou(pred_, gt_masks)
#
#         return loss_total
class RepirLoss(nn.Module):
    def __init__(self):
        super(RepirLoss, self).__init__()
        self.softiou = SoftIoULoss()
        # self.softiou = FocalSoftIoULoss()
        # self.softiou32 = SplitIoULoss(32)

        self.iou = IoULoss()
        #self.iou = FocalHardIoULoss1()

        self.l1 = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean',pos_weight=torch.tensor(10))
        #self.psloss = pixelsumLoss()
        self.bce_sum = nn.BCELoss(reduction='sum')
        # 512*512 --> 2*2
        self.pool2 = nn.MaxPool2d(kernel_size=256, stride=256)
        # 512*512 --> 4*4
        self.pool4 = nn.MaxPool2d(kernel_size=128, stride=128)
        # 512*512 --> 8*8   256*256 --> 4*4
        self.pool8 = nn.MaxPool2d(kernel_size=64, stride=64)
        # 512*512 --> 16*16 256*256 --> 8*8
        self.pool16 = nn.MaxPool2d(kernel_size=32, stride=32)
        self.down16 = nn.Upsample(scale_factor=1/32,mode='bilinear')
        #                   256*256 --> 16*16
        self.pool32 = nn.MaxPool2d(kernel_size=16, stride=16)
        #                   256*256 --> 32*32
        self.pool64 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.pool128 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool256 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer_lvl_num = 2

        self.threshold = 0.45

        #self.weighting = Aligned_MTL()
        self.grad = Get_gradient_nopadding()
        #self.vfocal = VFLoss()


        self.up2x = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        self.up4x = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16x = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        # self.point_loss = HungarianMatcher(num_points=200)
        # self.lovaze = LovaszSoftmax()
        self.focal = nn.BCELoss()


    def forward(self, pred, gt_masks):
        pred_, classify_result= pred
        b,_,_,_ = pred_.shape

        with torch.no_grad():
            classify_gt = self.pool256(gt_masks)
        loss_class = self.iou(classify_result,classify_gt)
        # loss_class = self.softiou(classify_result,gt_masks)
        # loss_mse = self.amse(pred_, gt_masks)
        # loss_point = self.point_loss(pred_, gt_masks)
        # loss_dice = dice_loss(pred_, gt_masks.flatten(1),1)


        loss_iou = self.iou(pred_, gt_masks)

        # loss_total = idx_epoch/400 * loss_iou + (400 - idx_epoch)/400 * loss_class
        loss_total =  loss_iou + 0.8 * loss_class


        return loss_total


def FocalIoULoss(inputs, targets):
    "Non weighted version of Focal Loss"

    # def __init__(self, alpha=.25, gamma=2):
    #     super(WeightedFocalLoss, self).__init__()
    # targets =
    # inputs = torch.relu(inputs)
    [b, c, h, w] = inputs.size()

    # inputs = torch.nn.Sigmoid()(inputs)
    inputs = 0.999 * (inputs - 0.5) + 0.5
    BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
    intersection = torch.mul(inputs, targets)
    smooth = 1

    IoU = (intersection.sum() + smooth) / (inputs.sum() + targets.sum() - intersection.sum() + smooth)

    alpha = 0.75
    gamma = 2
    num_classes = 2
    # alpha_f = torch.tensor([alpha, 1 - alpha]).cuda()
    # alpha_f = torch.tensor([alpha, 1 - alpha])
    gamma = gamma
    size_average = True

    pt = torch.exp(-BCE_loss)

    F_loss = torch.mul(((1 - pt) ** gamma), BCE_loss)

    at = targets * alpha + (1 - targets) * (1 - alpha)

    F_loss = (1 - IoU) * (F_loss) ** (IoU * 0.5 + 0.5)

    F_loss_map = at * F_loss

    F_loss_sum = F_loss_map.sum()

    return F_loss_map, F_loss_sum

class FIoULoss(nn.Module):
    def __init__(self):
        super(FIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            # 若列表或者元组，每一个计算loss并且取平均
            loss_total = 0
            for i in range(len(preds)):
                # 新添加修改
                # pred = torch.nn.Sigmoid()(preds[i])
                # pred = torch.sigmoid(preds[i])

                pred = preds[i]
                _,loss = FocalIoULoss(pred,gt_masks)
                # loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            # 新添加修改
            # pred = torch.nn.Sigmoid()(preds)
            # pred = torch.sigmoid(preds)
            pred = preds  # 其它方法的head 输出时已经进行过sigmoid
            smooth = 1
            # print(pred.shape)
            # print(gt_masks.shape)
            _,loss = FocalIoULoss(pred,gt_masks)
            # loss = 1 - loss.mean()
            return loss