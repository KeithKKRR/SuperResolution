import argparse

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from data.datasets import CelebA_HQ_Dataset
from models.SRCNN import SRCNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = vars(parser.parse_args())

    # read config
    with open("config/" + args["model"] + ".yml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    args.update(cfg)

    # dataset and dataloader
    test_dataset = CelebA_HQ_Dataset("data/test_data.txt", 64, 256)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    model = SRCNN()
    model.load_state_dict(torch.load("checkpoint/SRCNN.pth"))
    for (LR_img, HR_img) in test_dataloader:
        SR_img = model(LR_img)
        SR_img = transforms.ToPILImage()(SR_img[0])
        HR_img = transforms.ToPILImage()(HR_img[0])
        SR_img.save("output/test_SR.png")
        HR_img.save("output/test_GT.png")
        break
