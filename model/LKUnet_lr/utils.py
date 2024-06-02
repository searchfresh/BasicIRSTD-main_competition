from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

# 单卷积主干分支，用于消融实验
class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.main_branch2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=5,
                      stride=1,
                      padding=5 // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x1 = self.main_branch(x)
        x2 = self.main_branch2(x1)
        return x2


class conv_block_lk(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block_lk, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=11, stride=1, padding=5, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Sobel_x_Block(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(Sobel_x_Block, self).__init__()
        # 定义Sobel算子
        self.sobel_x = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 不可训练的参数
        self.sobel_x = nn.Parameter(self.sobel_x.repeat(in_channels, 1, 1, 1), requires_grad=False)
        # 添加适当的卷积层来改变通道数量
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)

        self.chan = in_channels

    def forward(self, x):
        # 计算水平方向上的Sobel卷积
        sobel_x_output = F.conv2d(x, self.sobel_x, padding=1,groups=self.chan)
        #使用卷积层改变通道数量
        sobel_x_output = self.conv(sobel_x_output)
        #print(sobel_x_output.shape)
        return sobel_x_output

class Sobel_y_Block(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(Sobel_y_Block, self).__init__()

        # 定义Sobel算子
        self.sobel_y = torch.tensor([[1, 2, 1],
                                     [0, 0, 0],
                                     [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 不可训练的参数
        self.sobel_y = nn.Parameter(self.sobel_y.repeat(in_channels, 1, 1, 1), requires_grad=False)
        # 添加适当的卷积层来改变通道数量
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.chan = in_channels

    def forward(self, x):
        # 计算垂直方向上的Sobel卷积
        sobel_y_output = F.conv2d(x, self.sobel_y, padding=1,groups=self.chan)
        # 使用卷积层改变通道数量
        sobel_y_output = self.conv(sobel_y_output)
        return sobel_y_output


class Laplacian_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Laplacian_Block, self).__init__()

        # 定义拉普拉斯算子
        self.laplacian_kernel = torch.tensor([[0, 1, 0],
                                              [1, -4, 1],
                                              [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 不可训练的参数
        self.laplacian_kernel = nn.Parameter(self.laplacian_kernel.repeat(in_channels, 1, 1, 1), requires_grad=False)

        # 添加适当的卷积层来改变通道数量
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.chan = in_channels

    def forward(self, x):
        # 计算拉普拉斯算子卷积
        laplacian_output = F.conv2d(x, self.laplacian_kernel, padding=1,groups=self.chan)

        # 使用卷积层改变通道数量
        laplacian_output = self.conv(laplacian_output)

        return laplacian_output


class Tophat_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tophat_Block, self).__init__()
        # # 定义结构元素
        # self.structure = torch.tensor([[0, 1, 0],
        #                              [1, 1, 1],
        #                              [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        #
        # # 不可训练的结构元素
        # self.structure = nn.Parameter(self.structure.repeat(in_channels, 1, 1, 1), requires_grad=False)

        self.conv3 = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        self.chan = in_channels

    def dilate(self,bin_img, ksize=5):
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
        out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
        return out

    def erode(self,bin_img, ksize=5):
        out = 1 - self.dilate(bin_img= 1 - bin_img, ksize=ksize)
        return out

    def forward(self, x):

        # 腐蚀操作
        erosion_result = self.erode(bin_img = x,ksize=3)

        # 膨胀操作
        dilation_result = self.dilate(bin_img=erosion_result,ksize=3)

        # Tophat 操作结果
        tophat_result = x - dilation_result

        # 修改通道数
        tophat_result = self.conv3(tophat_result)

        return tophat_result

class conv_block_lk_sobel(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block_lk_sobel, self).__init__()

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.sobel_x_branch = nn.Sequential(
            Sobel_x_Block(in_ch,out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.sobel_y_branch = nn.Sequential(
            Sobel_y_Block(in_ch,out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.L_branch = nn.Sequential(
            Laplacian_Block(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )



    def forward(self, x):
        # 主干网络的前向传播
        main_output = self.main_branch(x)

        # Sobel分支的前向传播
        sobel_x_output = self.sobel_x_branch(x)
        sobel_y_output = self.sobel_y_branch(x)

        #拉普拉斯分支的前向传播
        L_output = self.L_branch(x)


        # 将主干网络输出和Sobel分支的结果相加
        combined_output = main_output + sobel_x_output + sobel_y_output + L_output + x

        #print(combined_output.shape)
        return combined_output

class conv_block_lk_sobel_MoE(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block_lk_sobel_MoE, self).__init__()

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.main_branch2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=5,
                      stride=1,
                      padding=5//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.sobel_x_branch = nn.Sequential(
            Sobel_x_Block(in_ch,out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.sobel_y_branch = nn.Sequential(
            Sobel_y_Block(in_ch,out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.L_branch = nn.Sequential(
            Laplacian_Block(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.gate_num=3
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=in_ch,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(in_ch),
            nn.Tanh(),
            nn.Conv2d(in_ch,self.gate_num,1,1,0))
        # self.bn = nn.BatchNorm2d(out_ch)



    def forward(self, x):

        k = self.gate(self.gap(x)).squeeze()
        if len(k.shape)==1:
            k = k.unsqueeze(0)
        _, index = torch.max(k,dim=1)
        one_hot = F.one_hot(index,num_classes=self.gate_num)
        # softmax = F.softmax(k,dim=-1)
        # 主干网络的前向传播bushi
        main_output = self.main_branch(x)
        main_output = self.main_branch2(main_output)

        # Sobel分支的前向传播
        sobel_x_output = self.sobel_x_branch(x)
        sobel_y_output = self.sobel_y_branch(x)
        sobel = (sobel_y_output + sobel_x_output)/2
        #拉普拉斯分支的前向传播
        L_output = self.L_branch(x)

        out = torch.stack([main_output,sobel,L_output],dim=2)
        one_hot=one_hot.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(out)

        # 将主干网络输出和Sobel分支的结果相加
        combined_output = torch.sum(out*one_hot, dim=2)

        #print(combined_output.shape)
        return combined_output

class conv_block_lk_sobel_lap_TH_MoE(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block_lk_sobel_lap_TH_MoE, self).__init__()

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.main_branch2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=5,
                      stride=1,
                      padding=5//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.sobel_x_branch = nn.Sequential(
            Sobel_x_Block(in_ch,out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.sobel_y_branch = nn.Sequential(
            Sobel_y_Block(in_ch,out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.L_branch = nn.Sequential(
            Laplacian_Block(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.TH_branch = nn.Sequential(
            Tophat_Block(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.gate_num=4
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=in_ch,kernel_size=1,stride=1,padding=0),
            nn.BatchNorm2d(in_ch),
            nn.Tanh(),
            nn.Conv2d(in_ch,self.gate_num,1,1,0))
        # self.bn = nn.BatchNorm2d(out_ch)



    def forward(self, x):

        k = self.gate(self.gap(x)).squeeze()
        if len(k.shape)==1:
            k = k.unsqueeze(0)
        _, index = torch.max(k,dim=1)
        one_hot = F.one_hot(index,num_classes=self.gate_num)
        # softmax = F.softmax(k,dim=-1)
        # 主干网络的前向传播bushi
        main_output = self.main_branch(x)
        main_output = self.main_branch2(main_output)

        # Sobel分支的前向传播
        sobel_x_output = self.sobel_x_branch(x)
        sobel_y_output = self.sobel_y_branch(x)
        sobel = (sobel_y_output + sobel_x_output)/2
        #拉普拉斯分支的前向传播
        L_output = self.L_branch(x)

        #TopHat forward
        Tophat_output = self.TH_branch(x)
        out = torch.stack([main_output,sobel,L_output,Tophat_output],dim=2)
        one_hot=one_hot.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(out)

        # 将主干网络输出和Sobel分支的结果相加
        combined_output = torch.sum(out*one_hot, dim=2)

        #print(combined_output.shape)
        return combined_output

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x



class Unet_dict(nn.Module):
    """Unet which operates with filters dictionary values"""

    def __init__(self, n_labels, n_filters=32, p_dropout=0.5, batchnorm=True):
        super(Unet_dict, self).__init__()
        filters_dict = {}
        filt_pair = [3, n_filters]

        for i in range(4):
            self.add_module('contractive_' + str(i), ContractiveBlock(filt_pair[0], filt_pair[1], batchnorm=batchnorm))
            filters_dict['contractive_' + str(i)] = (filt_pair[0], filt_pair[1])
            filt_pair[0] = filt_pair[1]
            filt_pair[1] = filt_pair[1] * 2

        self.bottleneck = ConvolutionBlock(filt_pair[0], filt_pair[1], batchnorm=batchnorm)
        filters_dict['bottleneck'] = (filt_pair[0], filt_pair[1])

        for i in reversed(range(4)):
            self.add_module('expansive_' + str(i),
                            ExpansiveBlock(filt_pair[1], filters_dict['contractive_' + str(i)][1], filt_pair[0]))
            filters_dict['expansive_' + str(i)] = (filt_pair[1], filt_pair[0])
            filt_pair[1] = filt_pair[0]
            filt_pair[0] = filt_pair[0] // 2

        self.output = nn.Conv2d(filt_pair[1], n_labels, kernel_size=1)
        filters_dict['output'] = (filt_pair[1], n_labels)
        self.filters_dict = filters_dict

    # final_forward
    def forward(self, x):
        c00, c0 = self.contractive_0(x)
        c11, c1 = self.contractive_1(c0)
        c22, c2 = self.contractive_2(c1)
        c33, c3 = self.contractive_3(c2)
        bottle = self.bottleneck(c3)
        u3 = F.relu(self.expansive_3(bottle, c33))
        u2 = F.relu(self.expansive_2(u3, c22))
        u1 = F.relu(self.expansive_1(u2, c11))
        u0 = F.relu(self.expansive_0(u1, c00))
        return F.softmax(self.output(u0), dim=1)
