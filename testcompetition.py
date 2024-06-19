import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset3 import *
import os
from torch.optim.swa_utils import AveragedModel
from Tools.sliceInference import *
from filter_process import filter_large
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--model_names_1", default='LKUNet',
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--model_names_2", default='LKUNet2',
                    help="model_name_3: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--model_names_3", default='LKUNet3',
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")

parser.add_argument("--pth_dirs_1", default=r"./checkpoint/Dataset-mask/LKUNet13.pth",
                    help="checkpoint dir" )
parser.add_argument("--pth_dirs_2", default=r"./checkpoint/Dataset-mask/LKUNet_best17.pth",
                    help="checkpoint dir" )
parser.add_argument("--pth_dirs_3", default=r"./checkpoint/Dataset-mask/LKUNet_69.pth",
                    help="checkpoint dir" )

parser.add_argument("--model_names_4", default='HrisNet',
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--pth_dirs_4", default=r"./checkpoint/Dataset-mask/HrisNet_400_57.pth",
                    help="checkpoint dir" )

parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")

parser.add_argument("--dataset_names", default='WideIRSTD', type=str,
                    help="dataset_name: Dataset-mask 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUST'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.05)
parser.add_argument("--SWA", type=bool, default=True, help="Sliced for inference")
parser.add_argument("--filter_large", type=bool, default=True, help="Sliced for inference")

global opt
opt = parser.parse_args()


def test():
    test_set = TestSetLoader(opt.dataset_dir, opt.dataset_names, opt.dataset_names, None)
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

    net1 = Net(model_name=opt.model_names_1, mode='test').cuda()
    net2 = Net(model_name=opt.model_names_2, mode='test').cuda()
    net3 = Net(model_name=opt.model_names_3, mode='test').cuda()
    net4 = Net(model_name=opt.model_names_4, mode='test').cuda()
    if opt.SWA == True:
        net1 = AveragedModel(net1)
        net2 = AveragedModel(net2)
        # net3 = AveragedModel(net3)

    try:
        net1.load_state_dict(torch.load(opt.pth_dirs_1)['state_dict'], strict=True)
        net2.load_state_dict(torch.load(opt.pth_dirs_2)['state_dict'], strict=True)
        net3.load_state_dict(torch.load(opt.pth_dirs_3)['state_dict'], strict=True)
        net4.load_state_dict(torch.load(opt.pth_dirs_4)['state_dict'], strict=True)
    except:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net1.load_state_dict(torch.load(opt.pth_dirs_1, map_location=device)['state_dict'])
        net2.load_state_dict(torch.load(opt.pth_dirs_2, map_location=device)['state_dict'])
        net3.load_state_dict(torch.load(opt.pth_dirs_3, map_location=device)['state_dict'])
        net4.load_state_dict(torch.load(opt.pth_dirs_4, map_location=device)['state_dict'])
    net1.eval()
    net2.eval()
    net3.eval()
    net4.eval()

    with torch.no_grad():
        for idx_iter, (img, size, img_dir, ori_size) in enumerate(test_loader):
            img = Variable(img).cuda()
            # if size[0]>=4096 or size[1]>=4096:
            #     img = F.interpolate(input=img, size=(size[0]//4, size[0]//4), mode='bilinear', )
            # elif size[0]>=2048 or size[1]>=2048:
            #     img = F.interpolate(input=img, scale_factor=0.25, mode='bilinear',)
            if opt.filter_large:
                if size[-1] > 4096 or size[-2] > 4096:
                    # predits = np.array((pred[0, 0, :, :] > opt.threshold).cpu()).astype('int64')
                    # image_predict = measure.label(predits, connectivity=2)
                    # coord_image = measure.regionprops(image_predict)
                    # if len(coord_image) > 10:
                    #     pred = torch.zeros_like(pred)
                    # if size[0] > size[1]:
                    #     size_t = (1024, (int(1024 * torch.div(size[1], size[0], rounding_mode='floor')) // 2) * 2)
                    # else:
                    #     size_t = ((int(1024 * torch.div(size[0], size[1], rounding_mode='floor')) // 2) * 2, 1024)
                    #
                    # size_t = (2048,2048)
                    # img = F.interpolate(input=img, size=size_t, mode='bilinear', )
                    size_t = (4096, 4096)
                    img = F.interpolate(input=img, size=size_t, mode='bilinear', )
                    pred1 = net1.forward(img)
                    pred1 = F.interpolate(input=pred1, size=(size[0],size[1]),
                                        mode='bilinear', )
                    pred2 = net2.forward(img)
                    pred2 = F.interpolate(input=pred2, size=(size[0], size[1]),
                                            mode='bilinear', )
                    pred3 = net3.forward(img)
                    pred3 = F.interpolate(input=pred3[0], size=(size[0], size[1]),
                                            mode='bilinear', )

                    pred4 = slice_inference(img, size_t, 512, net4)
                    pred4 = F.interpolate(input=pred4, size=(size[0], size[1]),
                                          mode='bilinear', )
                else:
                    pred1 = net1.forward(img)
                    pred2 = net2.forward(img)
                    pred3 = net3.forward(img)
                    pred4 = net4.forward(img)
            else:
                pred1 = net1.forward(img)
                pred2 = net2.forward(img)
                pred3 = net3.forward(img)
                pred4 = slice_inference(img, size, 512, net4)


            if isinstance(pred1, list):
                pred1 = pred1[0]
            elif isinstance(pred1, tuple):
                pred1 = pred1[0]
            if isinstance(pred2, list):
                pred2 = pred2[0]
            elif isinstance(pred2, tuple):
                pred2 = pred2[0]
            if isinstance(pred3, list):
                pred3 = pred3[0]
            elif isinstance(pred3, tuple):
                pred3 = pred3[0]
            if isinstance(pred4, list):
                pred4 = pred4[0]
            elif isinstance(pred4, tuple):
                pred4 = pred4[0]


            # if size[0] >= 4096 or size[1] >= 4096:
            #     pred = F.interpolate(input=pred, size=(size[0], size[1]), mode='bilinear',)

            # pred = torch.maximum(pred1, pred2)
            # pred = torch.maximum(pred, pred3)
            #
            # # pred = pred[:, :, :size[0], :size[1]]
            # pred = pred[:, :, :ori_size[0], :ori_size[1]]


            if ori_size[-1] >= 1024 or ori_size[-2] >= 1024:
                pred1 = (pred1[:, :, :ori_size[0], :ori_size[1]] > opt.threshold).float()
                pred2 = (pred2[:, :, :ori_size[0], :ori_size[1]] > opt.threshold).float()
                pred3 = (pred3[:, :, :ori_size[0], :ori_size[1]] > opt.threshold).float()
                pred4 = (pred4[:, :, :ori_size[0], :ori_size[1]] > opt.threshold).float()
                pred = ((pred1 + pred2 + pred3 + pred4) > 1).float()
                pred = filter_large(pred.cpu())
            else:
                pred1 = (pred1[:, :, :ori_size[0], :ori_size[1]] > opt.threshold).float()
                pred2 = (pred2[:, :, :ori_size[0], :ori_size[1]] > opt.threshold).float()
                pred3 = (pred3[:, :, :ori_size[0], :ori_size[1]] > opt.threshold).float()
                pred4 = (pred4[:, :, :ori_size[0], :ori_size[1]] > opt.threshold).float()
                pred = ((pred1 + pred2 + pred3 + pred4) > 1).float()


            ### save img LKUNet
            if opt.save_img == True:
                img_save = transforms.ToPILImage()(pred[0, 0,:,:].cpu())
                if not os.path.exists(opt.save_img_dir + opt.dataset_names + '/' + opt.model_names_1):
                    os.makedirs(opt.save_img_dir + opt.dataset_names + '/' + opt.model_names_1)
                img_save.save(opt.save_img_dir + opt.dataset_names + '/' + opt.model_names_1 + '/' + img_dir[0] + '.png')


if __name__ == '__main__':
    if opt.pth_dirs_1 == None:
        raise RuntimeError
    else:
        test()