import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
from torchvision import transforms


Y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21]
intervals_256 = [(256, 512, 0), (512, 640, 1), (640, 768, 2), (768, 896, 3), (896, 1024, 4), (1024, 1152, 5),
         (1152, 1280, 6), (1152, 1280, 7), (1280, 1408, 8), (1408, 1536, 9), (1536, 1664, 10),
        (1664, 1792, 11), (1792, 1920, 12),(1920, 2049, 13),(2049,2176,14),(2176,2304,15),(2816,2944,16)]
intervals_384 = [(384, 768, 0), (768, 1152, 1), (1152, 1536, 2), (1536, 1920, 3), (1920, 2304, 4), (2304, 2688, 5),
         (2688, 3072, 6)]


def get_y_value(z, intervals, Y):
    """
    根据给定的值z，查找并返回Y数组中对应的值。

    参数:
    z (int or float): 需要查找的值。
    intervals (list of tuple): 定义区间的元组列表，每个元组包含三个元素，
                                分别是区间的下界（包含），区间的上界（不包含），和对应的Y索引。
    Y (list): 映射目标数组。

    返回:
    int or None: 对应的Y数组值，如果z不在任何区间内返回None。
    """
    for lower_bound, upper_bound, y_index in intervals:
        if lower_bound <= z < upper_bound:
            if y_index < len(Y):
                return Y[y_index]
            else:
                raise IndexError(f"Y数组索引 {y_index} 超出范围")
    return None  # 如果z不在任何区间内


def sliding_window_tensor(input_tensor, kernel_size, save_path=None, img_name=None):
    # 添加批次维度，并使用 unfold 获取滑动窗口
    input_tensor = input_tensor.float()
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)  # 形状变为 (1, C, H, W)
    # B, C, H, W = input_tensor.shape()
    image_height, image_width = input_tensor.shape[2], input_tensor.shape[3]

    if (image_height <= kernel_size) or (image_width <= kernel_size):
        # input_tensor = input_tensor.numpy().astype(np.uint8)
        if (image_width <= kernel_size) and (image_height <= kernel_size):
            resize_transforms = transforms.Resize((kernel_size ,kernel_size), interpolation=transforms.InterpolationMode.BILINEAR)
            input_tensor = resize_transforms(input_tensor)
        else:
            if (image_width <= kernel_size) and (image_height >= kernel_size):
                # input_tensor = np.resize(input_tensor, (image_height, kernel_size))
                resize_transforms = transforms.Resize((image_height, kernel_size),
                                                      interpolation=transforms.InterpolationMode.BILINEAR)
                input_tensor = resize_transforms(input_tensor)

            if (image_height <= kernel_size) and (image_width >= kernel_size):
                # input_tensor = np.resize(input_tensor, (kernel_size, image_width))
                resize_transforms = transforms.Resize((kernel_size, image_width),
                                                      interpolation=transforms.InterpolationMode.BILINEAR)
                input_tensor = resize_transforms(input_tensor)
        image_height = input_tensor.shape[2]
        image_width = input_tensor.shape[3]

    x = get_y_value(image_width, intervals_256, Y)
    y = get_y_value(image_height, intervals_256, Y)

    if image_height == kernel_size:
        stride_y = kernel_size
    else:
        stride_y = (image_height - kernel_size) // y  # 向下移动的距离
    if image_width == kernel_size:
        stride_x = kernel_size
    else:
        stride_x = (image_width - kernel_size) // x  # 向右移动的距离

    unfolded_tensor = F.unfold(input_tensor, kernel_size=(kernel_size, kernel_size), stride=(stride_y,stride_x ))
    B, C_H_W, num_patches = unfolded_tensor.shape  # 获取补丁的数量和通道数 unfold 输出是（B，C*K_H*K_W,L）
    unfolded_tensor = unfolded_tensor.transpose(2, 1)
    unfolded_tensor = unfolded_tensor.view(B*num_patches, 1, kernel_size,
                                           kernel_size).contiguous()  # 转换为:(B,num_pathcs,K_size,K_size)
    # unfolded_tensor = unfolded_tensor.squeeze(0)  # (num_patches,kernel_size, kernel_size)

    return unfolded_tensor,(stride_y,stride_x)

    ## 划分patch后保存各个patch
    # saved_image_paths = []
    # for idx in range(num_patches):
    #     # 提取每个滑动窗口
    #     window_tensor = unfolded_tensor[idx]
    #
    #     # 将张量转换为NumPy数组
    #     window_array = window_tensor.numpy().astype(np.uint8)
    #
    #     window_image = Image.fromarray(window_array.squeeze(), mode='L')
    #     # 将NumPy数组转换为PIL图像
    #     # if num_channels == 1:
    #     #     window_image = Image.fromarray(window_array.squeeze(), mode='L')
    #     # else:
    #     #     window_image = Image.fromarray(window_array, mode='RGB')
    #
    #     # 保存图像
    #     image_path = f"{save_path}/{img_name}_patch_{idx}.png"
    #     window_image.save(image_path)
    #     saved_image_paths.append(image_path)
    #
    # return saved_image_paths

    #####
    # 如果图像高度或宽度小于 patch_size，填充到至少一个 patch 大小
    # if image_height < kernel_size:
    #     padding_height = kernel_size - image_height
    # else:
    #     padding_height = (stride - (image_height - kernel_size) % stride) % stride
    #
    # if image_width < kernel_size:
    #     padding_width = kernel_size - image_width
    # else:
    #     padding_width = (stride - (image_width - kernel_size) % stride) % stride

    # 计算填充的高度和宽度
    # padding_height = (stride - (image_height - kernel_size) % stride) % stride
    # padding_width = (stride - (image_width - kernel_size) % stride) % stride
    # 进行填充
    # padded_image = F.pad(input_tensor, (0, padding_width, 0, padding_height))

    # unfolded_tensor = F.unfold(padded_image, kernel_size=(kernel_size, kernel_size), stride=stride)

    # B, C_H_W, num_patches = unfolded_tensor.shape # 获取补丁的数量和通道数 unfold 输出是（B，C*K_H*K_W,L）
    #
    # unfolded_tensor = unfolded_tensor.transpose(2, 1)
    # unfolded_tensor = unfolded_tensor.view(B, num_patches, kernel_size, kernel_size).contiguous() # 转换为:(B,num_pathcs,K_size,K_size)
    #
    # unfolded_tensor = unfolded_tensor.squeeze(0)  # (num_patches,kernel_size, kernel_size)
    #####
    # batch_size, num_channels_kernel_size, num_patches = unfolded_tensor.shape
    # num_channels = input_tensor.size(1)

    # 调整形状为 (num_patches, C, kernel_size, kernel_size)
    # unfolded_tensor = unfolded_tensor.permute(0, 2, 1)  # 形状变为 (batch_size, num_patches, num_channels * kernel_size * kernel_size)
    # unfolded_tensor = unfolded_tensor.view(batch_size, num_patches, num_channels, kernel_size, kernel_size)
    # unfolded_tensor = unfolded_tensor.permute(0, 4, 1, 2, 3).contiguous()  # 形状变为 (num_patches, C, kernel_size, kernel_size)
    # saved_image_paths = []

    # for idx in range(num_patches):
    #     # 提取每个滑动窗口
    #     window_tensor = unfolded_tensor[idx]
    #
    #     # 将张量转换为NumPy数组
    #     window_array = window_tensor.numpy().astype(np.uint8)
    #
    #     window_image = Image.fromarray(window_array.squeeze(), mode='L')
    #     # 将NumPy数组转换为PIL图像
    #     # if num_channels == 1:
    #     #     window_image = Image.fromarray(window_array.squeeze(), mode='L')
    #     # else:
    #     #     window_image = Image.fromarray(window_array, mode='RGB')
    #
    #     # 保存图像
    #     image_path = f"{save_path}/{img_name}_patch_{idx}.png"
    #     window_image.save(image_path)
    #     saved_image_paths.append(image_path)
    #
    # return saved_image_paths

if __name__ == '__main__':
    # 依照train.txt文件路径，提取并切片
    img_dir = r"D:\Dataset-mask-purify\dataset_split\dataset_1\train_mask(512)_before_slice"  # 图像所在文件夹
    save_dir = r"D:\Dataset-mask-purify\dataset_split\dataset_1\train_mask(512)_after_slice"  # 保存patch的文件夹
    train_txt = r"D:\Dataset-mask-purify\dataset_split\dataset_1\train.txt"  # train.txt文件路径
    # patch_size = 512
    # overlap = 0.1  # 指定重叠比例


    with open(train_txt, 'r') as file:
        image_names = [line.strip() for line in file]

        # 处理每张图像
    for img_name in image_names:
        # print(img_name)
        img_path = os.path.join(img_dir, f"{img_name}.png")
        # base_name = os.path.splitext(img_name)[0]

        # 预处理图像
        img = Image.open(img_path).convert("I")
        img = np.array(img)

        img_tensor = torch.tensor(img[np.newaxis, :])
        # img_tensor = torch.tensor(img).permute(2, 0, 1)  # 转换为张量并调整维度顺序为 (C, H, W)

        kernel_size = 512  # 滑动窗口大小
        save_path = save_dir  # 更新此路径为所需目录
        # 确保保存目录存在
        os.makedirs(save_path, exist_ok=True)
        saved_image_paths = sliding_window_tensor(img_tensor, kernel_size, save_path, img_name)

        # img = np.array(img)
        # img = load_image(img_path)
        # img = preprocess_image(img_path).unsqueeze(0)  # 添加批次维度 (B, C, H, W)
        #
        # 按重叠比例分割图像
        # patches, coords = image_to_patches(img, patch_size=patch_size, overlap=overlap)
        #
        # # 保存分割后的patch
        # save_patches(patches, coords, save_dir, base_name)

    print("图像分割并保存完成！")