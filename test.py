import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset3 import *
import matplotlib.pyplot as plt
from metrics import *
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--model_names", default=['LKUNet'], nargs='+',
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--pth_dirs", default=['./checkpoint/Dataset-mask/LKUNet_15.pth'], nargs='+',
                    help="checkpoint dir, default=None or ['NUDT-SIRST/ACM_400.pth.tar','NUAA-SIRST/ACM_400.pth.tar']")
parser.add_argument("--dataset_dir", default='D:\PycharmFile\BasicIRSTD-main\datasets', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['Dataset-mask'], nargs='+',
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()
## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
    opt.img_norm_cfg = dict()
    opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
    opt.img_norm_cfg['std'] = opt.img_norm_cfg_std


def test():
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, None)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=opt.model_name, mode='test').cuda()
    try:
        net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.load_state_dict(torch.load(opt.pth_dir, map_location=device)['state_dict'])
    net.eval()

    eval_mIoU = mIoU()
    eval_PD_FA = PD_FA()
    for idx_iter, (img, gt_mask, size, img_dir, ori_size) in enumerate(test_loader):
        img = Variable(img).cuda()
        pred = net.forward(img)
        if isinstance(pred, list):
            pred = pred[0]
        elif isinstance(pred, tuple):
            pred = pred[0]
        pred = pred[:, :, :ori_size[0], :ori_size[1]]
        gt_mask = gt_mask[:, :, :ori_size[0], :ori_size[1]]
        eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask)
        eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], ori_size)

        ### save img
        if opt.save_img == True:
            img_save = transforms.ToPILImage()((pred[0, 0, :, :]).cpu())
            if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name):
                os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name)
            img_save.save(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + img_dir[0] + '.png')

    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')


if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test' + '.txt', 'w')
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_400.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_400.pth.tar'
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    opt.test_data4set_name = dataset_name
                    opt.model_name = model_name
                    opt.train_dataset_name = pth_dir.split('/')[0]
                    print(pth_dir)
                    opt.f.write(pth_dir)
                    print(opt.test_dataset_name)
                    opt.f.write(opt.test_dataset_name + '\n')
                    # opt.pth_dir = opt.save_log + pth_dir
                    opt.pth_dir = pth_dir
                    test()
                    print('\n')
                    opt.f.write('\n')
        opt.f.close()