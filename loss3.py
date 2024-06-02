import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from torch.autograd import Variable
import torchvision.utils as vutils


# ----------------------------------------------------------------
# from LibMTL.weighting import Aligned_MTL as Aligned_MTL
# ----------------------------------------------------------------


class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            # 若列表或者元组，每一个计算loss并且取平均
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() - intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            # pred = torch.sigmoid(preds)
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() - intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss


class ISNetLoss(nn.Module):
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
        loss_edge = 10 * self.bce(preds[1], edge_gt) + self.softiou(preds[1].sigmoid(), edge_gt)

        return loss_img + loss_edge


def dice_loss(y_true, y_pred, smooth=1.0):
    intersection = torch.sum(y_true * y_pred)
    dice = (2.0 * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    loss = 1.0 - dice
    return loss.mean()


class BCEFocalLoss(nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.00001, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class pixelsumLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCELoss(reduction='mean')

    def forward(self, input, target):
        b, _, h, w = input.shape
        input = input.reshape(b, -1)
        target = target.reshape(b, -1)
        p_in = torch.sum(input, dim=1).cuda() / (torch.ones([b, 1]).cuda() * (h * w))
        p_ta = torch.sum(target, dim=1).cuda() / (torch.ones([b, 1]).cuda() * (h * w))
        loss = self.bceloss(p_in, p_ta)
        return loss


class SwinUnetLoss(nn.Module):
    def __init__(self):
        super(SwinUnetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        # self.bce = nn.BCELoss(reduction='mean')
        # self.psloss = pixelsumLoss()

    def forward(self, pred, gt_masks):
        b, _, _, _ = pred.shape
        ### img loss
        loss_img = self.softiou(pred, gt_masks)
        # loss_ps = self.psloss(pred, gt_masks)
        ### edge loss
        # print(loss_focal,loss_img + 3*loss_focal)
        # loss_bce = dice_loss(pred, gt_masks)
        # loss_bce = self.bce(pred.reshape(b,-1), gt_masks.reshape(b,-1))
        # print(loss_img, loss_bce)
        loss_dice = dice_loss(pred.reshape(b, -1), gt_masks.reshape(b, -1))
        return loss_img


from torch.utils.tensorboard import SummaryWriter
from mmcv.ops.point_sample import point_sample
from mmdet.models.utils.point_sample import get_uncertain_point_coords_with_randomness


class SearchNetLoss(nn.Module):
    def __init__(self,patch_size):
        super(SearchNetLoss, self).__init__()
        self.size = patch_size
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss(reduction='mean')
        # self.psloss = pixelsumLoss()
        self.bce_sum = nn.BCELoss(reduction='sum')
        # 512*512 --> 2*2
        self.pool2 = nn.MaxPool2d(kernel_size=patch_size//2, stride=patch_size//2)
        # 512*512 --> 4*4
        self.pool4 = nn.MaxPool2d(kernel_size=patch_size//4, stride=patch_size//4)
        # 512*512 --> 8*8   128*128 --> 2*2
        self.pool8 = nn.MaxPool2d(kernel_size=patch_size//8, stride=patch_size//8)
        # 512*512 --> 16*16 128*128 --> 4*4
        self.pool16 = nn.MaxPool2d(kernel_size=patch_size//16, stride=patch_size//16)
        #                   128*128 --> 8*8
        self.pool32 = nn.MaxPool2d(kernel_size=patch_size//32, stride=patch_size//32)
        #                   128*128 --> 16*16
        self.pool64 = nn.MaxPool2d(kernel_size=patch_size//64, stride=patch_size//64)

        self.pool128 = nn.MaxPool2d(kernel_size=patch_size//128, stride=patch_size//128)

        self.layer_lvl_num = 2

        self.threshold = 0.45

        # self.weighting = Aligned_MTL()
        self.grad = Get_gradient_nopadding()

        # self.writer = SummaryWriter('K:\\BasicIRSTD-main\\tensorboard')

    def forward(self, pred, gt_masks, idx_iter, idx_epoch):
        pred_, weighted_mask_1, weighted_mask_2, weighted_mask_3 = pred
        b, _, _, _ = pred_.shape
        loss_img = self.softiou(pred_, gt_masks)
        # print(pred_)
        loss_search = 0
        if self.size == 512:
            for i in range(self.layer_lvl_num-1):
                loss_search+=self.softiou(weighted_mask_1[i].reshape(b, -1),self.pool16(gt_masks).reshape(b, -1))
                loss_search += self.softiou(weighted_mask_2[i].reshape(b, -1), self.pool32(gt_masks).reshape(b, -1))
                loss_search += self.softiou(weighted_mask_3[i].reshape(b, -1), self.pool64(gt_masks).reshape(b, -1))
        else:
            for i in range(self.layer_lvl_num-1):
                loss_search+=self.bce(self.pool4(weighted_mask_1[i]).reshape(b, -1),self.pool2(gt_masks).reshape(b, -1))
                loss_search += self.bce(self.pool8(weighted_mask_2[i]).reshape(b, -1), self.pool4(gt_masks).reshape(b, -1))
                loss_search += self.bce(self.pool16(weighted_mask_3[i]).reshape(b, -1), self.pool8(gt_masks).reshape(b, -1))

        # vutils.save_image(self.pool64(weighted_mask_3[-1]), f"output/z_output_pool_{idx_iter}.png")
        loss_search /=3
        # local = torch.tensor(self.pool64(weighted_mask_3[-1])>self.threshold,dtype=float)
        local = self.pool16(gt_masks)
        mask = F.interpolate(
            local,
            pred_.shape[-2:],
            mode='bicubic',
            align_corners=True)

        masked_pred = torch.masked_select(pred_, mask > self.threshold)
        masked_gt = torch.masked_select(gt_masks, mask > self.threshold)

        # loss_local = self.bce(masked_pred, masked_gt)
        loss_local = self.bce(masked_pred, masked_gt)
        # vutils.save_image(torch.tensor(mask>self.threshold,dtype=float), f"output/z_output_mask_{idx_iter}.png")

        # edge
        edge_gt = self.grad(gt_masks.clone())

        ### edge loss
        loss_edge = self.softiou(self.grad(pred_.clone()), edge_gt)

        # weight = self.weighting(loss_img,loss_search,loss_local)
        # print(weight)

        if idx_epoch < 0:
            loss_total = loss_img + loss_search
        else:
            # print(loss_local)
            loss_total = loss_img + 0.2 * loss_search
            # loss_total = loss_img

        iter_ = idx_epoch * 80 + idx_iter + 1
        # self.writer.add_scalar("SoftIoULoss", loss_img,iter_)
        # self.writer.add_scalar("SearchLoss", loss_search, iter_)
        # self.writer.add_scalar("LocalLoss", loss_local, iter_)
        # self.writer.add_scalar("EdgeLoss", loss_edge, iter_)
        # self.writer.add_scalar("TotalLoss", loss_total, iter_)

        # if idx_epoch == 400:
        #     self.writer.close()
        return loss_total

