import random

import numpy as np

from utils import *
import matplotlib.pyplot as plt
import os
import albumentations
from VisualV1 import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        with open(self.dataset_dir + '/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()

        if img_norm_cfg == None:
            self.img_norm_cfg = None
        else:
            self.img_norm_cfg = img_norm_cfg
        self.mixup=False
        self.transform = albumentations.Compose([
            # albumentations.RandomScale((-0.1, 0.1),p=1),
            # albumentations.ShiftScaleRotate(p=1),
            albumentations.PadIfNeeded(patch_size,patch_size),
            albumentations.OneOf([albumentations.CropNonEmptyMaskIfExists(patch_size,patch_size,p=0.8),
                                  # albumentations.RandomCrop(patch_size,patch_size,p=0.2)
                                  ],p=1),

            # albumentations.RandomScale((-0.5,1.0),p=1),
            # albumentations.Resize(patch_size, patch_size),
            albumentations.OneOf([albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
                     albumentations.RandomRotate90(p=0.5),
                     albumentations.Flip(p=0.5),
                     albumentations.ElasticTransform(p=0.1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03), ], p=0.7),
            # albumentations.OneOf([albumentations.Sharpen(p=.5),
            #          albumentations.RandomBrightnessContrast(p=0.2), ]),
            albumentations.OneOf([
                albumentations.MotionBlur(p=.2),
                albumentations.MedianBlur(blur_limit=3, p=0.1),
                albumentations.Blur(blur_limit=3, p=0.1), ], p=0.3),
            # albumentations.MaskDropout(p=0.5)
            # albumentations.RandomResizedCrop(patch_size, patch_size),
            # albumentations.Transpose(p=0.5),
            # albumentations.HorizontalFlip(p=0.5),
            # albumentations.VerticalFlip(p=0.5),
            # albumentations.Rotate(p=0.5),
            # albumentations.HueSaturationValue(
            #     hue_shift_limit=0.2,
            #     sat_shift_limit=0.2,
            #     val_shift_limit=0.2,
            #     p=0.5
            # ),
            # albumentations.RandomBrightnessContrast(
            #     brightness_limit=(-0.1, 0.1),
            #     contrast_limit=(-0.1, 0.1),
            #     p=0.5
            # ),
            # albumentations.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            #     max_pixel_value=255.0,
            #     p=1.0
            # )
        ], p=1.)

    def __getitem__(self, idx):
        img_list = os.listdir(self.dataset_dir + '/images/')
        img_ext = os.path.splitext(img_list[0])[-1]
        if not img_ext in IMG_EXTENSIONS:
            raise TypeError("Unrecognized image extensions.")
        # print("读取的图片为：" + str(self.dataset_dir + '/images/' + self.train_list[idx] + img_ext))
        img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + img_ext).replace('//', '/')).convert('I')
        # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + img_ext).replace('//', '/'))

        mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + img_ext).replace('//', '/')).convert('I')

        # img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 3:
            mask = mask[:, :, 0]

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img_patch = transformed["image"]
            mask_patch = transformed["mask"]
        else:
            img_patch = img
            mask_patch = mask
        p=0
        if self.mixup:
            p = random.choice([0,1])
        if self.mixup and p==1:
            midx = random.randint(0,len(self.train_list)-1)
            mimg = Image.open(
                (self.dataset_dir + '/images/' + self.train_list[midx] + img_ext).replace('//', '/')).convert('I')
            mmask = Image.open(
                (self.dataset_dir + '/masks/' + self.train_list[midx] + img_ext).replace('//', '/')).convert('I')
            mimg = Normalized(np.array(mimg, dtype=np.float32), self.img_norm_cfg)
            mmask = np.array(mmask, dtype=np.float32) / 255.0
            if len(mmask.shape) > 3:
                mmask = mmask[:, :, 0]
            if self.transform is not None:
                mtransformed = self.transform(image=mimg, mask=mmask)
                mimg_patch = mtransformed["image"]
                mmask_patch = mtransformed["mask"]
            else:
                mimg_patch = mimg
                mmask_patch = mmask
            alpha = 10
            lam = np.random.beta(alpha,alpha)
            img_patch = lam*img_patch + (1-lam)*mimg_patch
            mask_patch = np.clip(pow(lam,0.5)*mask_patch+ pow((1-lam),0.5)*mmask_patch,0,1)

        # img_patch, mask_patch = random_crop(img, mask, self.patch_size)
        # img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis, :], mask_patch[np.newaxis, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch

    def __len__(self):
        return len(self.train_list)




class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, patch_size, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
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
        mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + img_ext).replace('//', '/')).convert('I')

        # img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        img = Normalized(np.array(img, dtype=np.float32), None)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 3:
            mask = mask[:, :, 0]


        img = PadImg(img, 32)
        mask = PadImg(mask, 32)

        # transform TTA
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        h, w = img.shape
        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)


class InferenceSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, patch_size, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(r"./test.txt", 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            # self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
            pass
        else:
            self.img_norm_cfg = img_norm_cfg

        self.transform = albumentations.Resize(patch_size, patch_size)

    def __getitem__(self, idx):
        img_list = os.listdir(self.dataset_dir + '/images/')
        img_ext = os.path.splitext(img_list[0])[-1]
        if not img_ext in IMG_EXTENSIONS:
            raise TypeError("Unrecognized image extensions.")
        # print("读取的图片为：" + str(self.dataset_dir + '/images/' + self.test_list[idx] + img_ext))
        # img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + img_ext).replace('//','/')).convert('I')
        img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + img_ext).replace('//', '/')).convert(
            'I')
        # mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + img_ext).replace('//','/'))
        mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + img_ext).replace('//', '/'))
        # img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        img = Normalized(np.array(img, dtype=np.float32), None)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 3:
            mask = mask[:, :, 0]

        h, w = img.shape
        # img = PadImg(img, 32)
        # mask = PadImg(mask, 32)

        # transform TTA
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h, w], self.test_list[idx]

    def __len__(self):
        return len(self.test_list)

class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        img_list_pred = os.listdir(self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/')
        img_ext_pred = os.path.splitext(img_list[0])[-1]

        img_list_gt = os.listdir(self.dataset_dir + '/masks/')
        img_ext_gt = os.path.splitext(img_list[0])[-1]

        if not img_ext_gt in IMG_EXTENSIONS:
            raise TypeError("Unrecognized GT image extensions.")
        if not img_ext_pred in IMG_EXTENSIONS:
            raise TypeError("Unrecognized Predicted image extensions.")
        mask_pred = Image.open((self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' +
                                self.test_list[idx] + img_ext_pred).replace('//', '/'))
        mask_gt = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + img_ext_gt).replace('//', '/'))

        mask_pred = np.array(mask_pred, dtype=np.float32) / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32) / 255.0
        if len(mask_pred.shape) > 3:
            mask_pred = mask_pred[:, :, 0]
        if len(mask_gt.shape) > 3:
            mask_gt = mask_gt[:, :, 0]

        h, w = mask_pred.shape

        mask_pred, mask_gt = mask_pred[np.newaxis, :], mask_gt[np.newaxis, :]

        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h, w]

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
