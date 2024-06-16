from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
from model import *
# from model.SwinTransformer.networkV3 import agpcnet
from skimage.feature.tests.test_orb import img
# from model.SearchNet.model_SearchNet import SearchNet
# from model.RepirDet.RepirDet_v2_t_256 import RepirDet
from loss2 import SearchNetLoss

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        
        self.cal_loss = SoftIoULoss()
        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')
        elif model_name == 'ABCNet':
            self.model = ABCNet()
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'LWNet':
            self.model = LWNet()
        elif model_name == 'ISNet':
            if mode == 'train':
                self.model = ISNet(mode='train')
            else:
                self.model = ISNet(mode='test')
            self.cal_loss = ISNetLoss()   #单独配置损失函数
        elif model_name == 'RISTDnet':
            self.model = RISTDnet()
        elif model_name == 'UIUNet':
            if mode == 'train':
                self.model = UIUNet(mode='train')
            else:
                self.model = UIUNet(mode='test')
        elif model_name == 'U-Net':
            self.model = Unet()
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()
        # elif model_name == 'SDS':
        #     self.model = SDS()
        elif model_name == 'SearchNet':
            self.model = SearchNet()
            self.cal_loss = SearchNetLoss()
        elif model_name == "MobileVit":
            self.model = MobileVit()
        elif model_name == 'RepirDet':
            if mode == 'train':
                self.model = RepirDet(deploy=False, mode='train')
            else:
                self.model = RepirDet(deploy=False, mode='test')
            self.cal_loss = RepirLoss()
        elif model_name == 'RepISD':
            if mode == 'train':
                self.model = RepISD()
            else:
                self.model = RepISD(deploy=True,convert=True)

            # only finetune
            # self.cal_loss = FIoULoss(

        elif model_name == "LKUNet":
            self.model = LKUNet()

        elif model_name == "LKUNet2":
            self.model = LKUNet2()

        elif model_name == "LKUNet3":
            self.model = LKUNet3()
            # only finetune
            # self.cal_loss = LKULoss()
        elif model_name == "HrisNet":
            self.model = HrisNet()

    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
