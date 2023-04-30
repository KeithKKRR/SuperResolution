import torch


def calculate_PSNR(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))
