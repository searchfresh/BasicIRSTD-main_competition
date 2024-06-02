# import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import matplotlib.pyplot as plt
import os
import math
import torch.nn as nn
from skimage import measure
import torch.nn.functional as F
import os
from torch.nn import init
import shutil
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def load_dataset (root, dataset, split_method):  # 为InferenceV2 所用
    train_txt = root + '/' + dataset + '/' + split_method + '/' + 'train.txt'
    test_txt  = root + '/' + dataset + '/' + split_method + '/' + 'test.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids,val_img_ids,test_txt

def total_visulization_generation(dataset_dir, mode:None, test_txt, suffix, target_image_path, target_dir): # 为InferenceV2 所用
    source_image_path = dataset_dir + '/images'

    txt_path = test_txt
    ids = []
    with open(txt_path, 'r') as f:
        ids += [line.strip() for line in f.readlines()]

    for i in range(len(ids)):
        source_image = source_image_path + '/' + ids[i] + suffix
        target_image = target_image_path + '/' + ids[i] + suffix
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + '/' + ids[i] + suffix
        img = Image.open(source_image)
        img = img.resize((256, 256), Image.ANTIALIAS)
        img.save(source_image)
    for m in range(len(ids)):
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + '/' + ids[m] + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Raw Imamge", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + '/' + ids[m] + '_GT' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + '/' + ids[m] + '_Pred' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Predicts", size=11)

        plt.savefig(target_dir + '/' + ids[m].split('.')[0] + "_fuse" + suffix, facecolor='w', edgecolor='red')

def save_Pred_GT(pred, labels, image_size:tuple, target_image_path, val_img_ids, num, suffix):  # 为InferenceV2 所用

    # predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    pred = pred.detach().cpu().numpy()
    pred = np.clip(pred, 0, 255)
    predsss = np.asarray(pred * 255, dtype=np.uint8)
    # predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(image_size[0], image_size[1]))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids) +suffix)
    # img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    # cv2.imwrite(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) + suffix, predsss)
    img = Image.fromarray(labelsss.reshape(image_size[0], image_size[1]))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids) + suffix)
    # img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)
    # cv2.imwrite(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix, labelsss)

def load_dataset (root, dataset): # 为InferenceV2 所用
    train_txt = root + '/' + dataset[0] + '/' + '/img_idx/train_' + dataset[0] + '.txt'
    test_txt  = root + '/' + dataset[0] + '/' + '/img_idx/test_' + dataset[0] + '.txt'

    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids,val_img_ids,test_txt

def seed_pytorch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 and classname.find('SplAtConv2d') == -1:
        init.xavier_normal(m.weight.data)
        
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
        
class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0
                
def random_crop(img, mask, patch_size):

    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size)-h),(0, max(w, patch_size)-w)), mode='constant')
        h, w = img.shape
    h_start = random.randint(0, h - patch_size)
    h_end = h_start + patch_size
    w_start = random.randint(0, w - patch_size)
    w_end = w_start + patch_size


    img_patch = img[h_start:h_end, w_start:w_end]
    mask_patch = mask[h_start:h_end, w_start:w_end]

    return img_patch, mask_patch

def Normalized(img, img_norm_cfg):
    return img/255.0
    # img = img/255.0

    # return (img-img_norm_cfg['mean'])/img_norm_cfg['std']

    
def Denormalization(img, img_norm_cfg):
    return img*img_norm_cfg['std']+img_norm_cfg['mean']

def get_img_norm_cfg(dataset_name, dataset_dir):
    if  dataset_name == 'NUAA-SIRST':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST':
        img_norm_cfg = dict(mean=107.80905151367188, std=33.02274703979492)
    elif dataset_name == 'IRSTD-1K':
        img_norm_cfg = dict(mean=87.4661865234375, std=39.71953201293945)
    elif dataset_name == 'NUDT-SIRST-Sea':
        img_norm_cfg = dict(mean=43.62403869628906, std=18.91838264465332)
    elif dataset_name == 'SIRST4':
        img_norm_cfg = dict(mean=62.10432052612305, std=23.96998405456543)
    elif dataset_name == 'IRDST-real':
        img_norm_cfg = dict(mean=101.54053497314453, std=56.49856185913086)
    elif dataset_name == None:
        img_norm_cfg = dict(mean=0, std=255)
    else:
        with open(dataset_dir+'/'+dataset_name+'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            train_list = f.read().splitlines()
        with open(dataset_dir+'/'+dataset_name+'/img_idx/test_' + dataset_name + '.txt', 'r') as f:
            test_list = f.read().splitlines()
        img_list = train_list + test_list
        img_dir = dataset_dir + '/' + dataset_name + '/images/'
        mean_list = []
        std_list = []
        for img_pth in img_list:
            try:
                img = Image.open((img_dir + img_pth).replace('//','/')+'.jpg').convert('I')
            except:
                try:
                    img = Image.open((img_dir + img_pth).replace('//','/')+'.png').convert('I')
                except:
                    img = Image.open((img_dir + img_pth).replace('//','/')+'.bmp').convert('I')
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), std=float(np.array(std_list).mean()))
    return img_norm_cfg

def get_optimizer(net, optimizer_name, scheduler_name, optimizer_settings, scheduler_settings):
    if optimizer_name == 'Adamw':
        optimizer = torch.optim.AdamW(filter(lambda p : p.requires_grad, net.parameters()), lr=optimizer_settings['lr'])
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'Adagrad':
        optimizer  = torch.optim.Adagrad(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'SGD':
        optimizer  = torch.optim.SGD(net.parameters(), lr=optimizer_settings['lr'],momentum=0.9,weight_decay=1e-4)

    if scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_settings['step'], gamma=scheduler_settings['gamma'])
    elif scheduler_name   == 'CosineAnnealingLR':
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['epochs'], eta_min=scheduler_settings['min_lr'])
    
    return optimizer, scheduler
        

def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h//times+1)*times-h),(0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0),(0, (w//times+1)*times-w)), mode='constant')
    return img


class SDS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 block_num = 1,
                 coord = True,
                 ):
        super(SDS, self).__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels,
                          in_channels,
                          3,1,1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(block_num)
        ])
        self.convs.append(
            nn.Sequential(
                nn.Conv2d(in_channels,
                          1,
                          1),
                # nn.Sigmoid()
            )
        )
        if coord:
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels+2,
                          out_channels,
                          3,1,1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        self.coord = coord
        self.addcoords = AddCoords()

    def forward(self, input):
        B,C,H,W = input.shape
        map = input
        for m in self.convs:
            map = m(map)
        # map = map.reshape([B, 1, -1]).softmax(dim=-1).reshape(B,1,H,W)
        map = self.keep_topk(map)
        if self.coord:
            input = self.addcoords(input)
            map_ = map.repeat([1, C+2, 1, 1])
            input = input[map_ == True].reshape([B, C+2, H // 2, W // 2])
        else:
            map_ = map.repeat([1,C,1,1])
            input = input[map_ == True].reshape([B, C, H // 2, W // 2])

        input = self.out_conv(input)
        return input, map

    def keep_topk(self, x, ratio=0.25):
        with torch.no_grad():
            feature_map = x.clone()
            B, _, H, W = feature_map.shape

            num_to_keep = int(ratio * H * W)
            flat_feature_map = feature_map.view(B, -1).clone()
            thresholds = torch.topk(flat_feature_map, num_to_keep, dim=1, largest=True,sorted=False).indices
                # 使用阈值将每个Batch中的数据分为两部分：大于阈值的保留，小于阈值的置零
            mask = torch.zeros_like(flat_feature_map)
            for i in range(B):
                mask[i][thresholds[i]] = True
            # mask = flat_feature_map > thresholds.view(-1, 1)
            # mask_tmp = flat_feature_map == thresholds.view(-1, 1)
                # flat_feature_map = flat_feature_map * mask.float()
                # # 恢复一维向量为原始特征图形状
                # feature_map = flat_feature_map.view(B, 1, H, W)

        return mask.reshape(B, 1, H, W)


class LSDS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 pixel=512,
                 block_num = 1,
                 coord = True,
                 ):
        super(LSDS, self).__init__()

        self.Linear1 = nn.Sequential(
                nn.Linear(pixel**2,
                          64),
                nn.LayerNorm(64),
                nn.ReLU(inplace=True),
            )
        self.Linear2 = nn.Sequential(
            nn.Linear(64,
                      pixel**2),
            nn.LayerNorm(pixel**2),
            nn.ReLU(inplace=True),
        )

        self.convs = nn.Sequential(
                nn.Conv2d(in_channels,
                          1,
                          1),
                # nn.Sigmoid()
            )

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        B,C,H,W = input.shape
        map = input
        map = self.Linear2(self.Linear1(map.reshape([B,C,-1]))).reshape([B,C,H,W])
        for m in self.convs:
            map = m(map)
        # map = map.reshape([B, 1, -1]).softmax(dim=-1).reshape(B,1,H,W)
        map = self.keep_topk(map)
        map_ = map.repeat([1,C,1,1])
        input = input[map_ == True].reshape([B,C,H//2,W//2])
        input = self.out_conv(input)
        return input, map

    def keep_topk(self, x, ratio=0.25):
        with torch.no_grad():
            feature_map = x.clone()
            B, _, H, W = feature_map.shape

            num_to_keep = int(ratio * H * W)
            flat_feature_map = feature_map.view(B, -1).clone()
            thresholds = torch.topk(flat_feature_map, num_to_keep, dim=1, largest=True,sorted=False).indices
                # 使用阈值将每个Batch中的数据分为两部分：大于阈值的保留，小于阈值的置零
            mask = torch.zeros_like(flat_feature_map)
            for i in range(B):
                mask[i][thresholds[i]] = True
            # mask = flat_feature_map > thresholds.view(-1, 1)
            # mask_tmp = flat_feature_map == thresholds.view(-1, 1)
                # flat_feature_map = flat_feature_map * mask.float()
                # # 恢复一维向量为原始特征图形状
                # feature_map = flat_feature_map.view(B, 1, H, W)

        return mask.reshape(B, 1, H, W)
def down_with_mask(x,mask):
    B, C, H, W = x.shape
    map_ = mask.repeat([1, C, 1, 1])
    x = x[map_ == True].reshape([B, C, H // 2, W // 2])
    return x

def restore(value,mask):
    B,C,H,W = value.shape
    out = torch.zeros([B, C, H*2, W*2]).to(value.device)
    mask = mask.repeat([1,C,1,1])
    out[mask==True] = value.reshape([-1])
    return out

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r
    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        return ret