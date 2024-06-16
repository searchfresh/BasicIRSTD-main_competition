import torch
import torchvision.transforms as T
# from torchvision.models.segmentation import fcn_resnet50
from PIL import Image
import numpy as np
import torch.nn.functional as F
from Tools.unfoldGetPatch import sliding_window_tensor
# from VisualV1 import visual
# outLayerFeature = {}  # 创建全局变量，存储需要可视化层级的featuremap
from torchvision import transforms

resize1 = transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BILINEAR)
resize2 = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR)

def infer_patch(model, patch):
    with torch.no_grad():
        b,_,_,_ = patch.shape
        pred1 = model.forward(patch[0 : b//2])
        if isinstance(pred1, list):
            pred1 = pred1[0]
        elif isinstance(pred1, tuple):
            pred1 = pred1[0]
        pred2 = model.forward(patch[b // 2:])
        if isinstance(pred2, list):
            pred2 = pred2[0]
        elif isinstance(pred2, tuple):
            pred2 = pred2[0]
        pred = torch.cat([pred1,pred2],dim=0)

    return pred

def pad_image(img, patch_size):  #已弃用，使用unfold代替
    # 检查图像，如果不够至少一个Patch的大小则对短边进行填充
    B, C, H, W = img.shape
    pad_h = max(0, patch_size - H)
    pad_w = max(0, patch_size - W)

    if pad_h > 0 or pad_w > 0:
        padding = (0, pad_w, 0, pad_h)  # 左右和上下的填充大小
        img = F.pad(img, padding, mode='constant', value=0)

    return img


def image_to_patches(img, patch_size, overlap):   #已弃用，使用unfold代替

    img = pad_image(img, patch_size)  # 先检查图像，如果不够至少一个Patch的大小则对短边进行填充
    _, _, h, w = img.shape
    step = int(patch_size * (1 - overlap))
    patches = []
    coords = []
    for i in range(0, h - patch_size + 1, step):
        for j in range(0, w - patch_size + 1, step):
            patch = img[:, :, i:i+patch_size, j:j+patch_size]
            if patch.shape[-1] == patch_size and patch.shape[-2] == patch_size:
                patches.append(patch)
                coords.append((i, j))
    return patches, coords


def patches_to_image(patches, coords, base_size, patch_size: int = 256):
    # step = int(patch_size * (1 - overlap))
    # num_batchs = patches[0].shape[0]
    # num_channels = patches[0].shape[1]
    # 初始化用于存储合并后图像的张量
    full_output = torch.zeros((1, 1, base_size[0], base_size[1]), dtype=torch.float32).cuda()
    # 初始化用于跟踪每个像素被覆盖次数的张量
    # coverage_count = torch.zeros((1, num_channels, full_size, full_size), dtype=torch.float32).cuda()
    # patches_concat = torch.concat(full_output,dim=1)

    z = 0
    for i in range(0, base_size[0]-patch_size + 1, coords[0]):  # coords[0] = stride_y
        for j in range(0, base_size[1]-patch_size + 1, coords[1]): # coords[1] = stride_x
            full_output[0, :, i:i + patch_size, j:j + patch_size] = torch.maximum(
                 full_output[0, :, i:i + patch_size, j:j + patch_size], patches[z, :, :, :])
            z = z+1

    # for patch, (i, j) in zip(patches, coords):
    #     full_output[:, :, i:i + patch_size, j:j + patch_size] = torch.max(
    #         full_output[:, :, i:i + patch_size, j:j + patch_size], patch)
    #     coverage_count[:, i:i + patch_size, j:j + patch_size] += 1
    # coverage_count[coverage_count == 0] = 1  # Avoid division by zero
    # full_output /= coverage_count
    return full_output
# 未添加重叠率（旧）
# def patches_to_image(patches, coords, full_size, patch_size=128):
#     num_channels = patches[0].shape[0]
#     full_output = torch.zeros((num_channels, full_size, full_size), dtype=torch.float32)
#     for patch, (i, j) in zip(patches, coords):
#         full_output[:, i:i+patch_size, j:j+patch_size] = patch
#     return full_output


def slice_inference(img, base_size:tuple, patch_size: int, model):
    # 把图按patch划分
    patches, coords = sliding_window_tensor(img, patch_size) #b,1,h,w
    # patches, coords = image_to_patches(img=img, patch_size=patch_size, overlap=overlap)

    # patches = resize1(patches)

    # 推理各个patch
    patched_outputs = infer_patch(model, patches)  # patches是list[] 列表形式 存储每个patch推理后的结果

    # concatenated_tensor = torch.cat(patched_outputs, dim=1)   # 需要把list列表中的各个patch 按照通道数进行concat后送入到visual函数
    # patched_outputs = resize2(concatenated_tensor)
    # outLayerFeature[f"{img_name}_Patch"] = concatenated_tensor
    # visual(None, outLayerFeature)

    # 把推理后的patch拼接成一个原图
    full_output_patches = patches_to_image(patched_outputs, coords, base_size, patch_size=patch_size)  # (1,256,256)

    # 可视化拼接后的img
    # outLayerFeature[f"{img_name}_Pred"] = full_output_patches  # 需要把list列表中的各个patch 按照通道数进行concat
    # visual(None, outLayerFeature)


    # 推理整个图像
    # full_img_output = infer_patch(model, img)
    return full_output_patches

def main():
    # model = load_model()
    # img_path = 'path_to_your_image.jpg'
    base_size = 512
    patch_size = 256

    # Preprocess the image
    img = preprocess_image(img_path, base_size=base_size)

    # Split image into patches
    patches, coords = image_to_patches(img, patch_size=patch_size)

    # Infer each patch
    patched_outputs = [infer_patch(model, patch) for patch in patches]

    # Reassemble the patches into a full image
    full_output_patches = patches_to_image(patched_outputs, coords, base_size, patch_size=patch_size)

    # Infer the whole image
    full_img_output = infer_patch(model, img)

    # Optional: Convert tensor to numpy for visualization or further processing
    full_output_patches_np = full_output_patches.squeeze(0).numpy()
    full_img_output_np = full_img_output.squeeze(0).numpy()

    # Post-process and save results if needed
    print("Patches reassembled output shape:", full_output_patches_np.shape)
    print("Whole image output shape:", full_img_output_np.shape)

if __name__ == '__main__':
    main()
