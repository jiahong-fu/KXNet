### By Fu to generate anisoblur s-stride LR image for testing
import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision

try:
    sys.path.append('..')
    from data.util import imresize
    import utils as util
except ImportError:
    pass

def generate_mod_LR_bic():
    # set parameters
    up_scale = 4
    mod_scale = 4
    # set data dir
    # sourcedir = "/home/jiahongfu/Documents/project/low_level_paper_code/SOTA/dataset/Set5_SR/Set5/image_SR4"
    # sourcedir = "/home/jiahongfu/Documents/project/low_level_paper_code/SOTA/dataset/Set14_SR/Set14/Set14_image_HR"
    # sourcedir = "/home/jiahongfu/Documents/project/low_level_paper_code/SOTA/dataset/BSD100_SR/BSD100_image_HR"
    sourcedir = "/home/jiahongfu/Documents/project/low_level_paper_code/SOTA/dataset/Urban100_SR/Urban100_image_HR"
    
    savedir = "/home/jiahongfu/Documents/project/KXNet/codes/data/testset/aniso_noise25/Urban100_direct"

    # load PCA matrix of enough kernel
    # print("load PCA matrix")
    # pca_matrix = torch.load(
    #     "../../pca_matrix.pth", map_location=lambda storage, loc: storage
    # )
    # print("PCA matrix shape: {}".format(pca_matrix.shape))

    for theta in [0.0, 45.0, 90.0, 135.0]:
        degradation_setting = {
            "random_kernel": False,
            # "code_length": 10,
            "ksize": 21, # 11 for 2, 15 for 3, 21 for 4
            "scale": up_scale,
            "cuda": True,
            "rate_iso": 0.0,     # 1.0 for iso, 0.0 for aniso
            "sigma_x": 2.0, 
            "sigma_y": 4.0,
            "theta": theta,
            "noise": True,
            "noise_high": 25.0,
        }

        # set random seed
        util.set_random_seed(0)

        saveHRpath = os.path.join(savedir, "HR", "x" + str(mod_scale))
        # saveLRpath = os.path.join(savedir, "LR", "x" + str(up_scale))
        # saveBicpath = os.path.join(savedir, "Bic", "x" + str(up_scale))
        saveLRblurpath = os.path.join(savedir, "LRblur", "x" + str(up_scale))
        saveKernelpath = os.path.join(savedir, "Kernel", "x" + str(up_scale))

        if not os.path.isdir(sourcedir):
            print("Error: No source data found")
            exit(0)
        if not os.path.isdir(savedir):
            os.mkdir(savedir)

        if not os.path.isdir(os.path.join(savedir, "HR")):
            os.mkdir(os.path.join(savedir, "HR"))
        # if not os.path.isdir(os.path.join(savedir, "LR")):
        #     os.mkdir(os.path.join(savedir, "LR"))
        # if not os.path.isdir(os.path.join(savedir, "Bic")):
        #     os.mkdir(os.path.join(savedir, "Bic"))
        if not os.path.isdir(os.path.join(savedir, "LRblur")):
            os.mkdir(os.path.join(savedir, "LRblur"))
        if not os.path.isdir(os.path.join(savedir, "Kernel")):
            os.mkdir(os.path.join(savedir, "Kernel"))

        if not os.path.isdir(saveHRpath):
            os.mkdir(saveHRpath)
        else:
            print("It will cover " + str(saveHRpath))

        # if not os.path.isdir(saveLRpath):
        #     os.mkdir(saveLRpath)
        # else:
        #     print("It will cover " + str(saveLRpath))

        # if not os.path.isdir(saveBicpath):
        #     os.mkdir(saveBicpath)
        # else:
        #     print("It will cover " + str(saveBicpath))

        if not os.path.isdir(saveLRblurpath):
            os.mkdir(saveLRblurpath)
        else:
            print("It will cover " + str(saveLRblurpath))

        if not os.path.isdir(saveKernelpath):
            os.mkdir(saveKernelpath)
        else:
            print("It will cover " + str(saveKernelpath))

        filepaths = sorted([f for f in os.listdir(sourcedir) if f.endswith(".png")])
        print(filepaths)
        num_files = len(filepaths)

        # kernel_map_tensor = torch.zeros((num_files, 1, 10)) # each kernel map: 1*10

        # prepare data with augementation
        
        for i in range(num_files):
            filename = filepaths[i]
            print("No.{} -- Processing {}".format(i, filename))
            # read image
            image = cv2.imread(os.path.join(sourcedir, filename))

            width = int(np.floor(image.shape[1] / mod_scale))
            height = int(np.floor(image.shape[0] / mod_scale))
            # modcrop
            if len(image.shape) == 3:
                image_HR = image[0 : mod_scale * height, 0 : mod_scale * width, :]
            else:
                image_HR = image[0 : mod_scale * height, 0 : mod_scale * width]
            # LR_blur, by random gaussian kernel
            img_HR = util.img2tensor(image_HR)
            C, H, W = img_HR.size()
            
            # for sig in np.linspace(1.35, 2.40, 8):

            prepro = util.SRMDPreprocessing(**degradation_setting)

            LR_img, ker_map = prepro(img_HR.view(1, C, H, W))
            image_LR_blur = util.tensor2img(LR_img)
            sigma_x = degradation_setting['sigma_x']
            sigma_y = degradation_setting['sigma_y']
            theta = degradation_setting['theta']
            noise = degradation_setting['noise_high'] if degradation_setting['noise'] else 0
            cv2.imwrite(os.path.join(saveLRblurpath, 'sig{}-{}_theta{}_noise{}_{}'.format(sigma_x,sigma_y,theta,noise,filename)), image_LR_blur)
            cv2.imwrite(os.path.join(saveHRpath, 'sig{}-{}_theta{}_noise{}_{}'.format(sigma_x,sigma_y,theta,noise,filename)), image_HR)

            # kernel_map_tensor[i] = ker_map

        image_LR_blur = util.tensor2img(LR_img)
        kernel = 1 / (np.max(util.tensor2img(ker_map.squeeze())) + 1e-4) * 255 * util.tensor2img(ker_map.squeeze())
        cv2.imwrite(os.path.join(saveKernelpath, 'sig{}-{}_theta{}_{}'.format(sigma_x,sigma_y,theta,filename)), kernel)
        # save dataset corresponding kernel maps
        # torch.save(kernel_map_tensor, './Set5_sig2.6_kermap.pth')
        print("Image Blurring & Down smaple Done: X{}, sig{}-{}_theta{}_noise{}".format(up_scale, sigma_x,sigma_y,theta, noise))


if __name__ == "__main__":
    generate_mod_LR_bic()