import torch


# from skimage.measure import compare_psnr as psnr
# from skimage.measure import compare_ssim as ssim

def calculate_PSNR(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

# def calculate_SSIM(im1, im2):
#     isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
#     s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
#              multichannel=isRGB)
#     return s
