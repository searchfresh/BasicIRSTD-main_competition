import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset3 import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from CalculateFPS import *
from utils import load_dataset, save_Pred_GT, total_visulization_generation
import torch.nn.functional as F
import torchvision.utils as vutils
import cv2
from torchstat import stat
from torchsummary import summary

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")

# =======================================model=========================================
parser.add_argument("--model_names", default=["LKUNet"], type=list,
                    help="model_name:  LKUNet RepirDet ,'DNANet', 'ALCNet','ACM', 'UIUNet', 'ISNet','ISTDU-Net'  'RDIAN', 'U-Net', 'RISTDnet',SwinTransformer")

# =======================================datasets============================
parser.add_argument("--dataset_names", default=['Dataset-mask'], type=list,
                    help="dataset_name: Dataset-mask 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUST'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")

# =====================================batchSize===========================================
parser.add_argument("--batchSize", type=int, default=1, help="Training batch sizse")
parser.add_argument("--patchsize", type=int, default=256, help="Training patch size")

# =====================================PTH===========================================
parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
# parser.add_argument("--resume", default=None, type=list, help="Resume from exisiting checkpoints (default: None)")
463
# parser.add_argument("--resume", default=["D:\PycharmFile\BasicIRSTD-main\log\Dataset-mask\LKUNet_178.pth"], type=list, help="Resume from exisiting checkpoints (default: None)")
# parser.add_argument("--resume", default=["checkpoint/SIRST3/DNANet_275.pth"], type=list, help="Resume from exisiting checkpoints (default: None)")
parser.add_argument("--resume", default=["D:\PycharmFile\BasicIRSTD-main\checkpoint\Dataset-mask\LKUNet_194_lr_ocr_psd_558_LOWFA.pth"], type=list, help="Resume from exisiting checkpoints (default: None)")

parser.add_argument("--deep_supervision", type=str, default=None, help="DSV (default: None)")

# ======================================Epoch==============================================
parser.add_argument("--nEpochs", type=int, default=400, help="Number of epochs")

#=======================================optimizer==========================================
parser.add_argument("--optimizer_name", default='Adam', type=str, help="Adagrad,Adam,SGD,optimizer name: Adam, Adagrad, SGD")

#==========================================lr=============================================
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.1}, type=dict,help="scheduler settings")

# =======================================
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=list, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=3407, help="Threshold for test")
parser.add_argument("--savemodel", type=str, default=False, help="Save model for best miou")
parser.add_argument("--tensorboard", type=str, default=False, help="Tensorboard for train and test")
parser.add_argument("--inferenceFPS", type=str, default=False, help="claculate FPS for inference")
parser.add_argument("--save_Pred_GT", type=str, default=False, help="save Pred_GT for inference")
parser.add_argument("--save_ori_prd_gt", type=str, default=False, help="use utils to save Pred_GT for inference")
parser.add_argument("--Calculate_Params", type=str, default=False, help="claculate Params for inference")
parser.add_argument("--sliced_pred", type=str, default=False, help="Sliced for inference")
parser.add_argument("--sliced_patch_size", type=int, default=256, help="Sliced for inference")
parser.add_argument("--saveThresholdInference", type=str, default=False, help="Sliced for inference")
global opt
opt = parser.parse_args()
seed_pytorch(opt.seed)
# current_file_path = os.path.abspath(__file__)
# current_file_name = os.path.basename(current_file_path)   # 获取当前运行的文件名称
now = datetime.now()
formatted_date = now.strftime("%Y-%m-%d_%H_%M_%S")
if opt.tensorboard:
    writer = SummaryWriter('log/Tensorboard/'+"allthreshold"+opt.model_names[0]+"_"+opt.dataset_names[0]+"_"+str(formatted_date))


class Trainer(object):
    def __init__(self):
        # train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchsize,
        #                            img_norm_cfg=opt.img_norm_cfg)
        # self.train_data = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, pin_memory=False)
        # dataset3 专用
        # test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, patch_size=opt.patchsize, img_norm_cfg=opt.img_norm_cfg)

        # dataset 专用
        test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name,patch_size=opt.patchsize, img_norm_cfg=opt.img_norm_cfg)
        self.test_data = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False, pin_memory=False)

        self.net = Net(model_name=opt.model_name, mode='train').cuda()
        print(f"推理的模型为:{opt.model_name}")
        self.epoch_state = 0
        self.total_loss_list = [0]
        self.total_loss_epoch = []
        self.best_miou = 0
        self.best_miou_epochs = 0
        self.best_info = (0, 0)

        # 获取要Inference的图片名称(已经被dataset中返回的图片名称代替)
        # self.train_img_ids, self.val_img_ids, self.test_txt = load_dataset("datasets", opt.dataset_names)
        if opt.resume:
            ckpt = torch.load(opt.resume[0])
            if opt.model_name == 'RepirDet':
                self.net.model.switch_to_deploy()
            self.net.load_state_dict(ckpt['state_dict'], strict=True)
            print("已加载预训练权重: " + opt.resume[0])
            # self.epoch_state = ckpt['epoch']
            # self.total_loss_list = ckpt['total_loss']
        else:
            print("未加载预训练权重")

        if opt.sliced_pred:
            print("使用切片推理")
            print("记得把dataset里面的resize注销掉哦！")
        else:
            print("使用正常推理")

    def test(self):
        self.net.eval()
        eval_mIoU = mIoU()
        eval_PD_FA = PD_FA()
        eval_nIoU = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        fps = 0
        time_ = 0

        if opt.Calculate_Params:
            # print(type(self.net.model))
            # print(self.net)
            # summary(self.net, (1, 224, 224))  # (C,H,W)
            summary(self.net, (1, 512, 512)) # (C,H,W)
            # x = torch.rand((1, 512, 512))
            # x = x.to('cuda')
            # stat(self.net, (3, 256, 256))  # 使用stat时需要把net.cuda()注释掉或者在cuda上创建测试用的tensor
            # stat(self.net, x)  # 使用stat时需要把net.cuda()注释掉或者在cuda上创建测试用的tensor

        if opt.model_name == 'RepirDet':
            self.net.model.switch_to_deploy()
        with torch.no_grad():
            # num = 0
            for idx_iter, (img, gt_mask, size, image_name) in enumerate(self.test_data):
                img = Variable(img).cuda()  # (1,1,512,512)
                gt_mask = Variable(gt_mask)
                if opt.sliced_pred:
                    # 这里调用 切片推理
                    pass
                else:
                    pred = self.net.forward(img)  # (1,1,512,512)
                if isinstance(pred, list):
                    pred = pred[0]
                elif isinstance(pred, tuple):
                    pred = pred[0]

                pred = pred[:, :, :size[0], :size[1]]
                gt_mask = gt_mask[:, :, :size[0], :size[1]]
                # 把处理后的ori，prd，gt拼接成一张图
                if opt.save_ori_prd_gt:
                    ori_pre_gt_location = f"inference/{opt.model_names[0]}/ori_pre_gt/"
                    ori_pre_gt = torch.concat([img, pred, gt_mask.cuda()], dim=0)
                    if not os.path.exists(ori_pre_gt_location):
                        os.makedirs(ori_pre_gt_location)
                    vutils.save_image(ori_pre_gt, ori_pre_gt_location+f"ori-pre-gt_{image_name[0]}.png", padding=5, pad_value=155)

                # 每张单独保存推理图，GT图
                if opt.save_Pred_GT:
                    Inference_result_Location = "inference/" + opt.model_names[0] + "/Inference_result"
                    if not os.path.exists(Inference_result_Location):
                        os.makedirs(Inference_result_Location)
                    save_Pred_GT(pred, gt_mask, (size[0], size[1]), Inference_result_Location, image_name[0], None, ".png")
                    # num += 1

                eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask)
                eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)
                eval_nIoU.update((pred > opt.threshold).cpu(), gt_mask)
            # total_visulization_generation(dataset_dir, args.mode, self.test_txt, ".png", self.visulization_path, self.visulization_fuse_path)

                # 为每张图片都单独输出PA，mIOU（记得修改metrics中Mioud 的get()函数最后添加self.reset()）
                # results1 = eval_mIoU.get()
                # results2 = eval_PD_FA.get()
                # results3 = eval_nIoU.get()
                # print(f"Image:{image_name[0]}, pixAcc, mIoU: " + str(results1))
                # print("PD, FA:\t" + str(results2))
                # print("nIoU:\t" + str(results3[1]))

                # 单独写入每张图片的miou
                # opt.f.write(f"Image:{image_name[0]}, pixAcc, mIoU:\t" + str(results1))
                # opt.f.write(f"Image:{image_name[0]}, PD, FA:\t" + str(results2) + '\n')
                # opt.f.write('\n')

            results1 = eval_mIoU.get()
            results2 = eval_PD_FA.get()
            results3 = eval_nIoU.get()

            if opt.inferenceFPS:
                # 新增(start)添加FPS
                fps = FPSBenchmark(
                    model=self.net,
                    device="cuda:0",
                    datasets=self.test_data,
                    iterations=self.test_data.__len__(),
                    log_interval=10
                ).measure_inference_speed()

        # 所有图片的总miou
        print("pixAcc, mIoU:\t" + str(results1))
        print("PD, FA:\t" + str(results2))
        print("nIoU:\t" + str(results3[1]))
        if opt.inferenceFPS:
            print("FPS:"+str(fps))
        if opt.saveThresholdInference:
            print("阈值 \t\t IOU \t\t PD \t\t FA \t\t NIou")
            print("%.1f \t\t%f \t%f \t\t%f  \t\t%f \n "%(opt.threshold, results1[1], results2[0], results2[1], results3[1]))
            opt.f.write(str(opt.threshold) + "\t\t" + str(results1[1]) +"\t\t" + str(results2[0])+"\t\t" + str(results2[1])+"\t\t"+ str(results3[1]) + '\n')
        # print(f"推理时间为：{time_:.4f}秒")

if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            if opt.saveThresholdInference:
                opt.f = open(opt.save + '/' + "allThreshold"+"_" + opt.dataset_name + '_' + opt.model_name + '_' + str(
                    formatted_date) + '.txt', 'w')
                opt.f.write("阈值 \t\t IOU \t\t PD \t\t FA \t\t NIou \n")
            # for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
                for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            # for threshold in [0.5]:
                    opt.threshold = threshold
                    print("数据集:" + opt.dataset_name + '_' + " 模型:" + opt.model_name + '_' +' 阈值为:' + str(opt.threshold))
                    trainer = Trainer()
                    trainer.test()
                print('\n')
                print('推理完成！')
                print("推理结果已保存至:" + opt.save + '/' + "allThreshold" + "_" + opt.dataset_name + '_' + opt.model_name + '_' + str(
                        formatted_date) + '.txt')
                opt.f.close()
            else:
                trainer = Trainer()
                trainer.test()
            print('\n')
            print('推理完成！')
        # opt.f.close()
