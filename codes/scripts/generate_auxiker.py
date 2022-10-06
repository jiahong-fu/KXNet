# by Fu to generate the auxiker map
# import img_utils
import cv2
import matplotlib.pyplot as plt
# # import torch.utils.tensorboard.SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F
import math
from scipy.io import savemat

############ generate auxiliary kernel map for iso setting
def stable_batch_kernel(batch, l=21, sig=2.6, tensor=True):
    sigma = sig
    ax = np.arange(-l // 2 + 1.0, l // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    xx = xx[None].repeat(batch, 0)
    yy = yy[None].repeat(batch, 0)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / np.sum(kernel, (1, 2), keepdims=True)
    return torch.FloatTensor(kernel) if tensor else kernel


load_map = torch.load("/home/jiahongfu/Downloads/temp/auxi_batch_ker.pth")

batch_ker = torch.zeros((24, 21, 21))
# i=0
# for sig in np.linspace(0.2, 2.0, 32):
#     print("sig:{}".format(sig))
#     batch_ker[i] = stable_batch_kernel(batch=1, l=21, sig=sig)
#     i += 1

i=0
for sig in np.linspace(0.2, 4.0, 24):
    print("sig:{}".format(sig))
    batch_ker[i] = stable_batch_kernel(batch=1, l=21, sig=sig)
    i += 1

torch.save(batch_ker, "../../auxi_batch_ker.pth")
############################################################################

############## generate auxiliary kernel map for aniso setting
def cal_sigma(sig_x, sig_y, radians):
    sig_x = sig_x.view(-1, 1, 1)
    sig_y = sig_y.view(-1, 1, 1)
    radians = radians.view(-1, 1, 1)

    D = torch.cat([F.pad(sig_x ** 2, [0, 1, 0, 0]), F.pad(sig_y ** 2, [1, 0, 0, 0])], 1)
    U = torch.cat([torch.cat([radians.cos(), -radians.sin()], 2),
                   torch.cat([radians.sin(), radians.cos()], 2)], 1)
    sigma = torch.bmm(U, torch.bmm(D, U.transpose(1, 2)))

    return sigma
def anisotropic_gaussian_kernel(batch, kernel_size, covar):
    ax = torch.arange(kernel_size).float() - kernel_size // 2

    xx = ax.repeat(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    yy = ax.repeat_interleave(kernel_size).view(1, kernel_size, kernel_size).expand(batch, -1, -1)
    xy = torch.stack([xx, yy], -1).view(batch, -1, 2)

    inverse_sigma = torch.inverse(covar)
    kernel = torch.exp(- 0.5 * (torch.bmm(xy, inverse_sigma) * xy).sum(2)).view(batch, kernel_size, kernel_size)

    return kernel / kernel.sum([1, 2], keepdim=True)

def stable_aniso_kernel(l=21, theta=0, sigma_x=0.2, sigma_y=4.0, tensor=True):
    theta = torch.ones(1) * theta / 180 * math.pi
    sigma_x = torch.ones(1) * sigma_x
    sigma_y = torch.ones(1) * sigma_y

    covar = cal_sigma(sigma_x, sigma_y, theta)
    kernel = anisotropic_gaussian_kernel(1, l, covar)
    return torch.FloatTensor(kernel) if tensor else kernel

# kernel_map = torch.zeros((24, 11, 11))    # For x2
# kernel_map = torch.zeros((24, 15, 15))    # For x3
# kernel_map = torch.zeros((24, 21, 21))    # For x4
# print(kernel_map.shape)
j=0
for sigma_x in [1.0, 2.5, 4.0]:
    for sigma_y in [3.5, 5.0]:
        for theta in np.linspace(0, 135, 4):
            kernel_map[j] = stable_aniso_kernel(l=11, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y)
            j += 1
torch.save(kernel_map, "../../auxi_batch_anisokerx2.pth")
###################################################################