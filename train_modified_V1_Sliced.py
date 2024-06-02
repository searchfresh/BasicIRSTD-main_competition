import argparse
import random
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
from SliceTest import *
from torchstat import stat   # 计算模型参数 方一
from torchsummary import summary  # 计算模型参数 方二
import torch.nn.functional as F
import torchvision.utils as vutils
import cv2
import albumentations

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")

# =======================================model=========================================
parser.add_argument("--model_names", default=["LKUNet"], type=list,
                    help="model_name:  LKUNet  RepISD MobileVit, SearchNet RepirDet ,'ACM', 'ALCNet', 'ISNet','RDIAN','DNANet' 'ISTDU-Net' 'UIUNet', , 'U-Net', 'RISTDnet',SwinTransformer")

# =======================================datasets============================
parser.add_argument("--dataset_names", default=['Dataset-mask'], type=list,
                    help="dataset_name: 'Dataset-mask','NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUST'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")

# =====================================batchSize===========================================
parser.add_argument("--batchSize", type=int, default=16, help="Training batch sizse")
parser.add_argument("--patchSize", type=int, default=448, help="Training patch size")

parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
# parser.add_argument("--resume", default=["D:\PycharmFile\BasicIRSTD-main\checkpoint\Dataset-mask\LKUNet_172_684.pth"], type=list, help="Resume from exisiting checkpoints (default: None)")
parser.add_argument("--resume", default=["D:\PycharmFile\BasicIRSTD-main\checkpoint\Dataset-mask\LKUNet_9_sd_717.pth"], type=list, help="Resume from exisiting checkpoints (default: None)")
# parser.add_argument("--resume", default=None, type=list, help="Resume from exisiting checkpoints (default: None)")
# parser.add_argument("--resume", default=["D:\PycharmFile\BasicIRSTD-main\checkpoint\Dataset-mask\LKUNet_194_lr_ocr_psd_64.pth"], type=list, help="Resume from exisiting checkpoints (default: None)")
# parser.add_argument("--resume", default=["log/Dataset-mask/RepirDet_166.pth"], type=list, help="Resume from exisiting checkpoints (default: None)")
# parser.add_argument("--resume", default=["checkpoint/IRSTD-1K/ALCNet_124.pth"], type=list, help="Resume from exisiting checkpoints (default: None)")
# parser.add_argument("--resume", default=["checkpoint/IRSTD-1K/ISNet_102.pth"], type=list, help="Resume from exisiting checkpoints (default: None)")
# parser.add_argument("--resume", default=["checkpoint/IRSTD-1K/RDIAN_26.pth"], type=list, help="Resume from exisiting checkpoints (default: None)")
# parser.add_argument("--resume", default=["checkpoint/IRSTD-1K/DNANet_130.pth"], type=list, help="Resume from exisiting checkpoints (default: None)")
# parser.add_argument("--resume", default=["checkpoint/IRSTD-1K/ISTDU-Net_300.pth"], type=list, help="Resume from exisiting checkpoints (default: None)")
# parser.add_argument("--resume", default=["checkpoint/IRSTD-1K/UIUNet_20.pth"], type=list, help="Resume from exisiting checkpoints (default: None)")
parser.add_argument("--deep_supervision", type=str, default=None, help="DSV (default: None)")

# ======================================Epoch==============================================
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs")
#=======================================optimizer==========================================
parser.add_argument("--optimizer_name", default='Adamw', type=str, help="Adagrad,Adam,SGD,optimizer name: Adam, Adagrad, SGD")

#==========================================lr=============================================
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.1}, type=dict,help="scheduler settings")

# =======================================
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
parser.add_argument("--savemodel", type=str, default=True, help="Save model for best miou")
parser.add_argument("--tensorboard", type=str, default=False, help="Tensorboard for train and test")
parser.add_argument("--inferenceFPS", type=str, default=False, help="claculate FPS for inference")
parser.add_argument("--orth", type=bool, default=False, help="Orthogonal Regularization")
parser.add_argument("--sliced_pred", type=str, default=False, help="Sliced for inference")
parser.add_argument("--sliced_patch_size", type=int, default=256, help="Sliced for inference")
parser.add_argument("--multiscale_training", type=bool, default=False, help="Sliced for inference")


global opt
opt = parser.parse_args()
seed_pytorch(opt.seed)
# current_file_path = os.path.abspath(__file__)
# current_file_name = os.path.basename(current_file_path)   # 获取当前运行的文件名称
now = datetime.now()
formatted_date = now.strftime("%Y-%m-%d_%H_%M_%S")
if opt.tensorboard:
    writer = SummaryWriter('log/Tensorboard/'+opt.model_names[0]+"_"+opt.dataset_names[0]+"_"+str(formatted_date))

class Trainer(object):
    def __init__(self):
        if opt.multiscale_training==True:
            patchSize_ = random.choice([opt.patchSize*6//4,])
        else:
            patchSize_ = opt.patchSize
        train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=patchSize_,
                                   img_norm_cfg=opt.img_norm_cfg)
        self.train_data = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, pin_memory=True)
        # dataset3 专用
        test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, patch_size=opt.patchSize, img_norm_cfg=opt.img_norm_cfg)

        # dataset 专用
        # test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg)
        self.test_data = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False, pin_memory=True)

        self.net = Net(model_name=opt.model_name, mode='train').cuda()
        self.epoch_state = 0
        self.total_loss_list = [0]
        self.total_loss_epoch = []
        self.best_miou = 0
        self.best_miou_epochs = 0
        self.best_info = (0, 0)
        if opt.resume:
            if opt.model_name == 'RepirDet':
                self.net.model.switch_to_deploy()
            ckpt = torch.load(opt.resume[0])
            self.net.load_state_dict(ckpt['state_dict'], strict=False)
            # self.epoch_state = ckpt['epoch']
            self.total_loss_list = ckpt['total_loss']
            print("已加载预训练权重: " + opt.resume[0])

        else:
            print("未加载预训练权重")

        # Default settings
        if opt.optimizer_name == 'Adam':
            opt.optimizer_settings = {'lr': 0.0005}
            opt.scheduler_name = 'MultiStepLR'
            opt.scheduler_settings = {'epochs': 400, 'step': [150, 300], 'gamma': 0.1}

        # Default settings of DNANet
        if opt.optimizer_name == 'Adagrad':
            opt.optimizer_settings['lr'] = 0.0005
            opt.scheduler_name = 'CosineAnnealingLR'
            opt.scheduler_settings['epochs'] = 1500
            opt.scheduler_settings['min_lr'] = 1e-3

        # if opt.model_name == 'RepirDet' and opt.optimizer_name == 'Adamw':
        if  opt.optimizer_name == 'Adamw':
            # opt.optimizer_settings = {'lr': 0.0007}
            opt.optimizer_settings['lr'] = 0.00015
            # opt.scheduler_name = 'MultiStepLR'
            # opt.scheduler_settings = {'epochs': 400, 'step': [80, 280], 'gamma': 0.3333}
            opt.scheduler_name = 'CosineAnnealingLR'
            # # # opt.scheduler_name = 'CyclicLR'
            opt.scheduler_settings['epochs'] = 200
            opt.scheduler_settings['min_lr'] = 0.00015//3
            # opt.scheduler_name = 'MultiStepLR'
            # opt.scheduler_settings = {'epochs': 400, 'step': [10, 20], 'gamma': 0.1}

        opt.nEpochs = opt.scheduler_settings['epochs']

        self.t = albumentations.RandomScale((-0.5, 1.0), p=1)

        self.optimizer, self.scheduler = get_optimizer(self.net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,opt.scheduler_settings)

    def train(self):
        self.net.train()
        opt.f.write('train_modified\ndatasets3\nnet\nloss\n' + "优化器: %s\nbatchsize:%d\npatchsize:%d\n初始学习率: %f\n调度器: %s\n阈值：%s\n" % (
        opt.optimizer_name, opt.batchSize, opt.patchSize, opt.optimizer_settings['lr'], opt.scheduler_name, opt.threshold))
        for idx_epoch in range(self.epoch_state, opt.nEpochs):
            tbar = tqdm(self.train_data)
            for idx_iter, (img, gt_mask) in enumerate(tbar):
                img, gt_mask = Variable(img).cuda(), Variable(gt_mask).cuda()

                # t = self.t(image = img,mask = gt_mask)
                # img = t["image"]
                # gt_mask = t["mask"]
                # scale = random.choice([1.0,2.0,1.5,1.75,1.25])
                # _,_,h_,w_ = img.shape
                # sizeh = int(h_*scale)
                # sizew = int(w_*scale)
                # img = F.interpolate(img,size = (sizeh,sizew),mode="bilinear")
                # gt_mask = F.interpolate(gt_mask,size = (sizeh,sizew),mode="nearest")

                pred = self.net.forward(img)
                if opt.model_name == "SearchNet":
                    loss = self.net.loss(pred, gt_mask, idx_iter, idx_epoch)
                else:
                    loss = self.net.loss(pred, gt_mask)

                # ori_pre_gt = torch.concat([img, pred, gt_mask], dim=0)
                # vutils.save_image(max_avg, f"features/opt.model_names/ori-gt_{idx_iter}.png", padding=5, pad_value=155)

                # if opt.deep_supervision == "DSV":
                #     preds = self.net.forward(img)
                #     loss = 0
                #     for pred in preds:
                #         loss += self.net.loss(pred, gt_mask)
                #     loss /= len(preds)
                # elif opt.deep_supervision == "None":
                #     pred = self.net.forward(img)
                #     loss = self.net.loss(pred, gt_mask)
                # total_loss_epoch.append(loss.detach().cpu()) #修改 1
                if opt.orth == True:
                    with torch.enable_grad():
                        reg = 1e-6
                        orth_loss = torch.zeros(1, device=loss.device)
                        param_flat_1 = None
                        param_flat_2 = None
                        param_flat_3 = None
                        param_flat_4 = None
                        for name, param in self.net.named_parameters():
                            if 'bias' and 'norm' not in name:
                                if 'stage1.0.branch_conv_list' in name:
                                    param_flat = param.view(param.shape[0], -1)
                                    if param_flat_1 is not None:
                                        param_flat_1 = torch.cat([param_flat_1, param_flat], dim=0)
                                    else:
                                        param_flat_1 = param_flat
                                elif 'stage2.0.branch_conv_list' in name:
                                    param_flat = param.view(param.shape[0], -1)
                                    if param_flat_2 is not None:
                                        param_flat_2 = torch.cat([param_flat_2, param_flat], dim=0)
                                    else:
                                        param_flat_2 = param_flat
                                elif 'stage3.0.branch_conv_list' in name:
                                    param_flat = param.view(param.shape[0], -1)
                                    if param_flat_3 is not None:
                                        param_flat_3 = torch.cat([param_flat_3, param_flat], dim=0)
                                    else:
                                        param_flat_3 = param_flat
                                elif 'stage4.0.branch_conv_list' in name:
                                    param_flat = param.view(param.shape[0], -1)
                                    if param_flat_4 is not None:
                                        param_flat_4 = torch.cat([param_flat_4, param_flat], dim=0)
                                    else:
                                        param_flat_4 = param_flat

                        sym1 = torch.mm(param_flat_1, torch.t(param_flat_1))
                        sym1 -= torch.eye(param_flat_1.shape[0], device=sym1.device)
                        sym2 = torch.mm(param_flat_2, torch.t(param_flat_2))
                        sym2 -= torch.eye(param_flat_2.shape[0], device=sym2.device)
                        sym3 = torch.mm(param_flat_3, torch.t(param_flat_3))
                        sym3 -= torch.eye(param_flat_3.shape[0], device=sym3.device)
                        sym4 = torch.mm(param_flat_4, torch.t(param_flat_4))
                        sym4 -= torch.eye(param_flat_4.shape[0], device=sym4.device)
                        orth_loss = (reg * sym1.abs().sum()) + (reg * sym2.abs().sum()) + (reg * sym3.abs().sum()) + (
                                    reg * sym4.abs().sum())
                    loss += orth_loss.item() / 4

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.total_loss_epoch.append(loss.item())  # 添加  1

                # tbar.set_description("Epoch:%3d, lr:%f, total_loss:%f" %(idx_epoch+1, optimizer.param_groups[0]['lr'], total_loss_list[-1]))    # 修改 2
                tbar.set_description("Epoch:%3d/%3d, lr:%f, total_loss:%f" % (
                idx_epoch + 1, opt.nEpochs, self.optimizer.param_groups[0]['lr'], self.total_loss_list[-1]))  # 添加 2
            self.scheduler.step()

            if (idx_epoch + 1) % 1 == 0:
                self.total_loss_list.append(float(np.array(self.total_loss_epoch).mean()))
                if opt.tensorboard:
                    writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], idx_epoch + 1)
                    writer.add_scalar("loss", self.total_loss_list[-1], idx_epoch + 1)
                # print(time.ctime()[4:-5] + ' Epoch---%d/%d, total_loss---%f'
                #       % (idx_epoch + 1, opt.nEpochs, total_loss_list[-1]))
                opt.f.write(time.ctime()[4:-5] + ' lr---%f, Epoch---%d/%d, total_loss---%f\n'
                            % (self.optimizer.param_groups[0]['lr'], idx_epoch + 1, opt.nEpochs, self.total_loss_list[-1]))
                self.total_loss_epoch = []

            if (idx_epoch + 1) % 1 == 0:
                # start_time = time.time()
                self.test(idx_epoch)
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(f"推理时间为：{elapsed_time:.4f}秒")
        if opt.tensorboard:
            writer.close()


    def test(self, idx_epoch):
        if opt.sliced_pred:
            print("使用切片推理")
        else:
            print("使用正常推理")
        self.net.eval()
        eval_mIoU = mIoU()
        eval_PD_FA = PD_FA()
        eval_nIoU = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        eval_nIoU.reset()
        time_ = 0
        # if opt.model_name == 'RepirDet':
        #     self.net.model.switch_to_deploy()
        with torch.no_grad():
            for idx_iter, (img, gt_mask, size, _) in enumerate(self.test_data):
                img = Variable(img).cuda()
                gt_mask = Variable(gt_mask)
                # start_time = time.time()
                if opt.sliced_pred:
                    # 这里调用 切片推理
                    pred = slice_inference(img, base_size=size, patch_size=opt.sliced_patch_size, model=self.net)
                else:
                    pred = self.net.forward(img)  # (1,1,512,512)
                # end_time = time.time()
                # time_ = time_ + (end_time - start_time)
                # vutils.save_image(torch.cat([img.cuda(), pred[0].cuda(), gt_mask.cuda()], dim=0), "output_test/"+ opt.model_names[0] + "_" + opt.dataset_names[0] + f"/ori_pred_gt{idx_iter}.png", padding=5, pad_value=155)

                if isinstance(pred, list):
                    pred = pred[0]
                elif isinstance(pred, tuple):
                    pred = pred[0]
                pred = pred[:, :, :size[0], :size[1]]
                # pred = pred[0][:, :, :size[0], :size[1]]
                gt_mask = gt_mask[:, :, :size[0], :size[1]]
                # 这里的pred没有加sigmoid，因此这边的值是原始值大于阈值，是错误的
                eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask)
                eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)
                eval_nIoU.update((pred > opt.threshold).cpu(), gt_mask)

            results1 = eval_mIoU.get()
            results2 = eval_PD_FA.get()
            results3 = eval_nIoU.get()
            temp_miou = results1[1]
        if temp_miou > self.best_miou:
            self.best_miou = temp_miou
            self.best_miou_epochs = idx_epoch + 1
            self.best_info = (self.best_miou, self.best_miou_epochs)
            if opt.savemodel:
                save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(
                    idx_epoch + 1) + '.pth'
                self.save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': self.net.state_dict(),
                    'total_loss': self.total_loss_list,
                }, save_pth)
        if opt.tensorboard:
            writer.add_scalar("pixAcc", results1[0], idx_epoch+1)
            writer.add_scalar("miou", results1[1], idx_epoch+1)
            writer.add_scalar("PD", results2[0], idx_epoch+1)
            writer.add_scalar("FA", results2[1], idx_epoch+1)
        print("pixAcc, mIoU:\t" + str(results1) + "\t best_miou, best_miou_epoch:" + str(self.best_info))
        print("PD, FA:\t" + str(results2))
        print("nIoU:\t" + str(results3[1]))
        opt.f.write("pixAcc, mIoU:\t" + str(results1) + "best_miou, best_miou_epoch:" + str(self.best_info) + '\n')
        opt.f.write("PD, FA:\t" + str(results2) + '\n')
        opt.f.write("nIoU:\t" + str(results3[1]) + '\n')
        opt.f.write('\n')
        print(f"推理时间为：{time_:.4f}秒")

        if opt.inferenceFPS and (idx_epoch + 1) % 5 == 0:
            # 新增(start)添加FPS
            FPSBenchmark(
                model=self.net,
                device="cuda:0",
                datasets=self.test_data,
                iterations=self.test_data.__len__(),
                log_interval=10
            ).measure_inference_speed()
            # 新增(end)
        self.net.train()


    def save_checkpoint(self, state, save_path):
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(state, save_path)
        return save_path

if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            print(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + str(formatted_date) + '.txt')
            # opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace( ':', '_') + '.txt', 'w')
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name + '_' + str(formatted_date) + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            trainer = Trainer()
            trainer.train()
            print('\n')
            opt.f.close()
