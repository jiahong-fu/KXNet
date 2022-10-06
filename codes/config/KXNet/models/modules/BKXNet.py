import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import scipy.io as io
import cv2

import sys
import utils as util
import matplotlib.pyplot as plt


############### torch.nn.functional.unfold ########################
def im2col(input, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2
    l = kernel_size
    s = stride
    # If a 4-tuple, uses(padding_left , padding_right , padding_top , padding_bottom)
    pad_data = torch.nn.ReflectionPad2d(padding=(pad, pad, pad, pad))
    input_padded = pad_data(input)
    col_input = torch.nn.functional.unfold(input_padded, kernel_size=(l, l),stride=(s, s))
    return col_input

def BC_im2col(input, channel, kernel_size, stride=1):
    x = input
    x_col = im2col(input=x, kernel_size=kernel_size, stride=stride)
    B, C_ks_ks, hw = x_col.size()
    x_col = x_col.reshape(B, channel, kernel_size*kernel_size, hw)
    x_col = x_col.permute(0, 2, 1, 3)
    x_col = x_col.reshape(B, kernel_size*kernel_size, channel*hw)
    return x_col


################## Bacthblur ################## 
#### 验证过是对的，但是输入的kernel是：B x feild_h x feild_w (或 feild_h x feild_w)
#### 打算通过stride来控制缩放尺度，试了一下好像行
class BatchBlur(object):
    def __init__(self, l=11):
        self.l = l
        if l % 2 == 1:
            self.pad =(l // 2, l // 2, l // 2, l // 2)
        else:
            self.pad = (l // 2, l // 2 - 1, l // 2, l // 2 - 1)
        # self.pad = nn.ZeroPad2d(l // 2)

    def __call__(self, input, kernel, stride=1):
        B, C, H, W = input.size()
        pad = F.pad(input, self.pad, mode='reflect')
        H_p, W_p = pad.size()[-2:]
        h = (H_p - self.l) // stride + 1
        w = (W_p - self.l) // stride + 1

        # if len(kernel.size()) == 2:
        #     input_CBHW = pad.view((C * B, 1, H_p, W_p))
        #     kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
        #     return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, h, w))
        # else:
        input_CBHW = pad.view((1, C * B, H_p, W_p))
        kernel_var = (
            kernel.contiguous()
            .view((B, 1, self.l, self.l))
            .repeat(1, C, 1, 1)
            .view((B * C, 1, self.l, self.l))
        )
        return F.conv2d(input_CBHW, kernel_var, stride=stride, groups=B * C).view((B, C, h, w))     # groups=B*C以后，相当于是逐channel

##############################################################

#################### Batch conv_transpose2d ##################
class BatchTranspose(object):
    def __init__(self, l=11):
        self.l = l
        self.pad = (self.l - 1) // 2

    def __call__(self, input, kernel, stride=1, output_padding=0):
        B, C, h, w = input.size()
        a = output_padding
        input_CBhw = input.view((1, B * C, h, w))
        H = (h - 1) * stride + self.l + a - 2 * self.pad
        W = (w - 1) * stride + self.l + a - 2 * self.pad
        kernel = (kernel.contiguous()
                    .view(B, 1, self.l, self.l)
                    .repeat(1, C, 1, 1)
                    .view(B * C, 1, self.l, self.l))

        return F.conv_transpose2d(input_CBhw, kernel, stride=stride, padding=self.pad, output_padding=a, groups=B * C).view(B, C, H, W)
###################################################################

#################### Inverse of PixelShuffle ######################
def inver_PS(x, r):
    [B, C, H, W] = list(x.size())
    x = x.reshape(B, C, H//r, r, W//r, r)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(B, C*(r**2), H//r, W//r)
    return x

#################### PixelShuffle #########################

## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# XNet Proximal operator
class X_ProNet(nn.Module):
    def __init__(self, scale):
        super(X_ProNet, self).__init__()
        if scale==2:
            self.channels = 59
        elif scale==3:
            self.channels=89
        else:   # x4
            self.channels=131
        self.resx1 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                                #    nn.BatchNorm2d(self.channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                                #    nn.BatchNorm2d(self.channels),
                                #    CALayer(self.channels),
                                   )
        self.resx2 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                                #   nn.BatchNorm2d(self.channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride =1, padding=1, dilation=1),
                                #   nn.BatchNorm2d(self.channels),
                                #   CALayer(self.channels)
                                  )
        self.resx3 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                                #   nn.BatchNorm2d(self.channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(self.channels, self.channels,  kernel_size=3, stride=1, padding=1, dilation=1),
                                #   nn.BatchNorm2d(self.channels),
                                #   CALayer(self.channels),
                                  )
        self.resx4 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1,  dilation=1),
                                #   nn.BatchNorm2d(self.channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1,  dilation=1),
                                #   nn.BatchNorm2d(self.channels),
                                #   CALayer(self.channels),
                                  )
        self.resx5 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1))

    def forward(self, input):
        x1 = F.relu(input + 0.1 * self.resx1(input))
        x2 = F.relu(x1 + 0.1 * self.resx2(x1))
        x3 = F.relu(x2 + 0.1 * self.resx3(x2))
        x4 = F.relu(x3 + 0.1 * self.resx4(x3))
        x5 = F.relu(input + 0.1 * self.resx5(x4))
        return x5


# KNet Proximal operator
class K_ProNet(nn.Module):
    def __init__(self, channels):
        super(K_ProNet, self).__init__()
        self.channels = channels

        self.resk1 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                                #    nn.BatchNorm2d(self.channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                                #    nn.BatchNorm2d(self.channels),
                                   )
        self.resk2 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                                #   nn.BatchNorm2d(self.channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride =1, padding=1, dilation=1),
                                #   nn.BatchNorm2d(self.channels),
                                  )
        self.resk3 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                                #  nn.BatchNorm2d(self.channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(self.channels, self.channels,  kernel_size=3, stride=1, padding=1, dilation=1),
                                #   nn.BatchNorm2d(self.channels),
                                  )
        self.resk4 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1,  dilation=1),
                                #   nn.BatchNorm2d(self.channels),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1,  dilation=1),
                                #   nn.BatchNorm2d(self.channels),
                                  )
        self.resk5 = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1))        


    def forward(self, input):
        k1 = F.relu(input + 0.1 * self.resk1(input))
        k2 = F.relu(k1 + 0.1 * self.resk2(k1))
        k3 = F.relu(k2 + 0.1 * self.resk3(k2))
        k4 = F.relu(k3 + 0.1 * self.resk4(k3))
        k5 = F.relu(input + 0.1 * self.resk5(k4))
        return k5


###############################

class XNet(nn.Module):
    def __init__(self, upscale=2, kernel_size=11):
        super(XNet, self).__init__()
        self.X_ProNet = X_ProNet(scale=upscale)
        self.BatchBlur = BatchBlur(l=kernel_size)
        self.BatchTranspose = BatchTranspose(l=kernel_size)

    def forward(self, x, AV_X, y, k, sf, eta):
        ##### Done 这里也考虑换成Bxhxw的kernel作为输入
        N, C, H, W = x.size() 
        k_B, k_h, k_w = k.size() 
        pad = (k_h - 1) // 2
        transpose_out_padding = (H + 2*pad - k_h) % sf
        E = self.BatchBlur(input=x, kernel=k, stride=sf) - y
        # print('E.min():{}, E.max():{}'.format(E.min(), E.max()))
        G_t = self.BatchTranspose(input=E, kernel=k, stride=sf, output_padding=transpose_out_padding)
        # print('G_t.max:{}, G_t.min:{}, G_t.shape():{}'.format(G_t.max(), G_t.min(), G_t.shape))
        #####################   E_ones for uneven overlap ################
        E_ones = torch.ones_like(E)
        G_ones_t = self.BatchTranspose(input=E_ones, kernel=k, stride=sf, output_padding=transpose_out_padding)
        # print('G_ones_t.max():{}, G_ones_t.min():{}'.format(G_ones_t.max(), G_ones_t.min()))
        # print("k.max():{}, k.min():{}".format(k.max(), k.min()))
        G_t = G_t / (G_ones_t + 1e-10)  
        ###################################################################

        # if torch.any(torch.isnan(G_t)):
        #     print("G_t maybe Nan!")
        #     print("G_t.max:{}, G_t.min:{}".format(G_t.max(), G_t.min()))
        #     torch.save(G_ones_t, "/home/iid/FJH/1/G_ones_t.pth")
        #     torch.save(G_t, "/home/iid/FJH/1/G_t.pth")
        # print('E.shape:{}, E_ones.shape:{}'.format(E.shape, E_ones.shape))

        # G = x - eta/100 * G_t    # Remarking for direct concat
        x_con = inver_PS(x, sf) # 3*sf*sf
        G_t_con = inver_PS(G_t, sf)
        # AV_X_con = inver_PS(AV_X, sf)   # 12*sf*sf
        PS = nn.PixelShuffle(sf)

        G_auxi = torch.cat((x_con, G_t_con, y, AV_X), dim=1)    # AV_X is auxiliary variable
        X_est_auxi = self.X_ProNet(G_auxi)
        X_est = PS(X_est_auxi[:, :3*sf*sf, :, :])
        # AV_X = PS(X_est_auxi[:, 2*(3*sf*sf)+3:, :, :])
        AV_X = X_est_auxi[:, 2*(3*sf*sf)+3:, :, :]
        # print("X_est.max:{}, X_est.min:{}".format(X_est.max(), X_est.min()))
        return X_est, AV_X

# class XNet(nn.Module):
#     def __init__(self):
#         super(XNet, self).__init__()
#         self.X_ProNet = X_ProNet(channels=35)
#         self.BatchBlur = BatchBlur()
#         self.BatchTranspose = BatchTranspose()

#     def forward(self, x, AV_X, y, k, sf, eta):
#         ##### Done 这里也考虑换成Bxhxw的kernel作为输入
#         N, C, H, W = x.size() 
#         k_B, k_h, k_w = k.size() 
#         pad = (k_h - 1) // 2
#         transpose_out_padding = (H + 2*pad - k_h) % sf
#         E = self.BatchBlur(input=x, kernel=k, stride=sf) - y

#         G_t = self.BatchTranspose(input=E, kernel=k, stride=sf, output_padding=transpose_out_padding)
#         # print('G_t.max:{}, G_t.min:{}, G_t.shape():{}'.format(G_t.max(), G_t.min(), G_t.shape))
#         for i in range(len(G_t)):
#             savepath = os.path.join( "/home/wqza/Documents/FJH/KXNet/codes/data/G_t_img/{:d}.png".format(i))
#             print('G_t_img.dim:', G_t[0].shape)
#             G_t_img = util.tensor2img(G_t[i, :, :, :].detach().float().cpu())
#             print('G_t_img.shape', G_t_img.shape)
#             util.save_img(G_t_img, savepath)
#         # print('x.shape:', x.shape)
#         G = x - eta * G_t
#         G_auxi = torch.cat((G, AV_X), dim=1)    # AV_X is auxiliary variable
#         X_est_auxi = self.X_ProNet(G_auxi)
#         X_est = X_est_auxi[:, :3, :, :]
#         AV_X = X_est_auxi[:, 3:, :, :]
#         return X_est, AV_X

class KNet(nn.Module):
    def __init__(self, kernel_size):
        super(KNet, self).__init__()
        self.K_ProNet = K_ProNet(channels=27)
        self.BatchBlur = BatchBlur(l=kernel_size)

    def forward(self, x, y, k, AV_k, sf, gamma):
        # x: NxCxHxW
        # K^(t+1/2)
        N, C, H, W = x.size()
        N_y, C_y, h, w = y.size()
        k_B, k_h, k_w, = k.size()  
        pad = (k_h - 1) // 2
        x_col_trans = BC_im2col(input=x, channel=C, kernel_size=k_h, stride=sf) # x_col_trans:(B, kxk, Cxhxw)
        # x_col = x_col_trans.permute(0,2,1)  # permute(0,2,1) x_col:(B, Cxhxw, kxk)
        # k = k.view(k_B, k_h*k_w, 1)
        # y_t = y.view(N_y, C_y*h*w, 1)
        # print('x_col.shape:', x_col.shape)
        # print('k.shape:', k.shape)
        # print('torch.matmul(x_col, k).shape:', torch.matmul(x_col, k).shape)
        # print('y_t.shape:', y_t.shape)
        # R = torch.matmul(x_col, k.view(k_B, k_h*k_w,1)) - y_t    # R:(B, Cxhxw, 1)
        R = (self.BatchBlur(input=x, kernel=k, stride=sf) - y).view(N_y, C_y*h*w, 1)
        G_K = (1 / (C_y*h*w)) * torch.matmul(x_col_trans, R)   # G_K:(B, kxk, 1)
        G_K = k.view(k_B, k_h*k_w,1) - gamma/10 * G_K
        G_K = torch.unsqueeze(G_K.reshape(k_B, k_h, k_w), dim=1).repeat(1, 3, 1, 1)

        C_auxi = torch.cat((G_K, AV_k), dim=1)    # AV_k is auxiliary variable
        # C_auxi = torch.cat((torch.unsqueeze(k.reshape(k_B, k_h, k_w), dim=1), AV_k), dim=1)   # only Prox
        K_temp1_auxi = self.K_ProNet(C_auxi)   # K^(t+1/2)
        K_temp1 = torch.mean(K_temp1_auxi[:, :3, :, :], dim=1, keepdim=True)
        # K_temp1 = K_temp1_auxi[:, :1, :, :]
        AV_k = K_temp1_auxi[:, 3:, :, :]

        ################ K_est normalize projection #################
        # K_temp1 = torch.squeeze(K_temp1).reshape(k_B, k_h*k_w, 1)

        # # K^(t+1)
        # I = torch.ones((k_B, k_h*k_w, 1)).cuda()
        # I_trans = I.permute(0,2,1)
        # K_temp2 = (torch.matmul(I_trans, K_temp1) - 1) / (k_h * k_w)
        # # print('I.shape:', I.shape)
        # # print('K_temp2.shape:', K_temp2.shape)
        # K_est = K_temp1 - torch.mul(K_temp2, I)
        # K_est = K_est.reshape(k_B, k_h, k_w)

        ########### K_est normalize K_est - K_est.min() / K_est.max() - K_est.min() #################
        # K_est_temp = torch.zeros_like(K_temp1)
        # for i in range(len(K_temp1)):
        #     K_est_temp[i] = ((K_temp1[i] - K_temp1[i].min()) / (K_temp1[i].max() - K_temp1[i].min())) / ((K_temp1[i] - K_temp1[i].min()) / (K_temp1[i].max() - K_temp1[i].min())).sum()
        #     # K_est_temp[i] = K_est_temp[i] / K_est_temp[i].sum()
        # K_est = K_est_temp.squeeze(dim=1)
        ########## K_est normalize torch.clamp ##############
        # K_est_temp = torch.clamp(K_temp1, 0, 0.999)
        # K_est = (K_est_temp / torch.sum(K_est_temp, dim=[2, 3], keepdim=True)).squeeze(dim=1)

        ######### K_est normalize F.relu #############
        K_est_temp = F.relu(K_temp1 + 1e-5)
        K_est = (K_est_temp / torch.sum(K_est_temp, dim=[2, 3], keepdim=True)).squeeze(dim=1)        

        return K_est, AV_k

######################################

##### Auxiliary Variable's kernel ######
AV_X_ker_def = (torch.FloatTensor([[0.0,1.0,0.0],[1.0,-4.0,1.0],[0.0,1.0,0.0]])).unsqueeze(dim=0).unsqueeze(dim=0)
# AV_X_ker_def = (torch.FloatTensor([[0.0,0.0,0.0],[0.0,1.0,0.0],[0.0,1.0,0.0]])).unsqueeze(dim=0).unsqueeze(dim=0)

# main Network architecture

class KXNet(nn.Module):
    def __init__(self, upscale, s_iter, 
                    kernel_size, ker_auxi_path=None):
        super(KXNet, self).__init__()
        self.iter = s_iter
        self.ksize = kernel_size
        # self.K_Net = KNet()
        # self.X_Net = XNet()
        self.scale = upscale
        self.X_stage = self.Make_XNet(self.iter)
        self.K_stage = self.Make_KNet(self.iter)

        # Auxiliary Variable
        self.AV_X_ker0 = AV_X_ker_def.expand(32, 3, -1, -1)
        self.AV_X_ker = nn.Parameter(self.AV_X_ker0, requires_grad=True)
        self.kernel_map0 = nn.Parameter(torch.load(ker_auxi_path), requires_grad=False) # [1, 441, 10] or [24, 21, 21]
        self.kernel_map = nn.Parameter(self.kernel_map0, requires_grad=True)
        #######################
        self.gamma_k = torch.Tensor([1.0])
        self.eta_x = torch.Tensor([1.0])
        self.gamma_stage = self.Make_Para(self.iter, self.gamma_k)
        self.eta_stage = self.Make_Para(self.iter, self.eta_x)

        # self.upscale = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
        #     nn.PixelShuffle(2),
        #     nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        # )

    def Make_XNet(self, iters):
        layers = []
        for i in range(iters):
            layers.append(XNet(self.scale, self.ksize))
        return nn.Sequential(*layers)
    
    def Make_KNet(self, iters):
        layers = []
        for i in range(iters):
            layers.append(KNet(self.ksize))
        return nn.Sequential(*layers)

    def gaussian_kernel_2d(self, kernel_size, sigma):
        kx = cv2.getGaussianKernel(kernel_size,sigma)
        ky = cv2.getGaussianKernel(kernel_size,sigma)
        return np.multiply(kx,np.transpose(ky)) 

    
    def Make_Para(self, iters, para):
        para_dimunsq = para.unsqueeze(dim=0)
        para_expand = para_dimunsq.expand(iters, -1)
        para = nn.Parameter(data=para_expand, requires_grad=True)
        return para

    # def forward(self, x, sf):
    def forward(self, x):
        srs = []
        kernel = []
        # srs_init = []
        
        B, C, H, W = x.shape
        # initialization and preparation calculation
        X_est = nn.functional.interpolate(input=x, scale_factor=self.scale, mode='bicubic', align_corners=False)
        # X_est = self.upscale(x)
        # srs_init.append(X_est)

        k_est = torch.as_tensor(self.gaussian_kernel_2d(kernel_size=self.ksize, sigma=1.0), dtype=torch.float32)

        # k_est = self.Make_kernel_ini(h_k=self.ksize, w_k=self.ksize, ratio=sf, mode='bicubic')
        # print('Init_k_est.min:{}, Init_k_est.max:{}, Init_sum_k:{}'.format(torch.min(k_est), torch.max(k_est), torch.sum(k_est)))
        k_est = k_est.unsqueeze(dim=0).repeat(B, 1, 1).cuda()
        # print('Init_k_est.min:{}, Init_k_est.max:{}, Init_sum_k:{}'.format(torch.min(k_est), torch.max(k_est), torch.sum(k_est)))
        ##### Done #######
        # AV_X = F.conv2d(X_est, self.AV_X_ker, stride=1, padding=1)  # Auxiliary Variable
        AV_X = F.conv2d(x, self.AV_X_ker, stride=1, padding=1)  # Auxiliary Variable
        # AV_k = self.kernel_map.permute(1,0).reshape(10, self.ksize, self.ksize).unsqueeze(dim=0)
        AV_k = self.kernel_map.unsqueeze(dim=0)
        AV_k = AV_k.repeat(B, 1, 1, 1)

        
        for i in range(self.iter):
            # print("################ stage:{} ##################".format(i))
            k_est, AV_k = self.K_stage[i](X_est, x, k_est, AV_k, self.scale, self.gamma_stage[i])
            # k_est = k
            # print("k_est.shape in loop:{}".format(k_est.shape))
            # k_est = k
            X_est, AV_X = self.X_stage[i](X_est, AV_X, x, k_est, self.scale, self.eta_stage[i])
            # print('stage:{}, gamma:{}'.format(i, self.gamma_stage[i]))
            # print('stage:{}, eta:{}'.format(i, self.eta_stage[i]))
            # print('k_est.min:{}, k_est.max:{}, k_est.sum:{}'.format(torch.min(k_est), torch.max(k_est), torch.sum(k_est)))
            # print('X_est.min:{}, X_est.max:{}'.format(torch.min(X_est), torch.max(X_est)))
            # print("kernel shape:{}".format(k_est.shape))
            srs.append(X_est)
            kernel.append(k_est)

        # return [srs_init, srs, kernel]
        return [srs, kernel]
