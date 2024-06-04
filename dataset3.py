import random

import numpy as np

from utils import *
import matplotlib.pyplot as plt
import os
# import albumentations


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')



class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, patch_size, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
        # with open(self.dataset_dir + '/img_idx/test' + '.txt', 'r') as f:
        # with open(r"D:\PycharmFile\BasicIRSTD-main\datasets\Dataset-mask\img_idx\test_Dtatset-mask.txt", 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            # self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
            pass
        else:
            self.img_norm_cfg = img_norm_cfg

        self.transform = None  # 使用切片推理时，打开此语句，保留图片原始尺寸
        # self.transform = albumentations.Resize(patch_size, patch_size)  # 使用正常推理时，打开此语句

    def __getitem__(self, idx):
        img_list = os.listdir(self.dataset_dir + '/images/')
        img_ext = os.path.splitext(img_list[0])[-1]
        if not img_ext in IMG_EXTENSIONS:
            raise TypeError("Unrecognized image extensions.")
        # print("读取的图片为：" + str(self.dataset_dir + '/images/' + self.test_list[idx] + img_ext))
        # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + img_ext).replace('//','/')).convert('I')


        # img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + img_ext).replace('//', '/')).convert('I')
        img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + img_ext).replace('//', '/')).convert('I')
        # mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + img_ext).replace('//','/'))
        # mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + img_ext).replace('//', '/')).convert('I')

        # img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        img = Normalized(np.array(img, dtype=np.float32), None)
        # mask = np.array(mask, dtype=np.float32) / 255.0
        # if len(mask.shape) > 3:
        #     mask = mask[:, :, 0]

        ori_h , ori_w = img.shape

        img = PadImg(img, 32)
        # mask = PadImg(mask, 32)

        # transform TTA
        # if self.transform is not None:
        #     transformed = self.transform(image=img, mask=mask)
        #     img = transformed["image"]
        #     mask = transformed["mask"]
        h, w = img.shape
        img = img[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        # mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, [h, w], self.test_list[idx], [ori_h , ori_w]

    def __len__(self):
        return len(self.test_list)



class augumentation(object):
    def __call__(self, input, target):
        if random.random() < 0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target
