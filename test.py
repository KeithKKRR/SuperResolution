import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.datasets import CelebA_HQ_Dataset
from models.model_utils import initialize_model_and_optimizer
from utils.device import device
from utils.logger import AverageMeter
from utils.metrics import calculate_PSNR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = vars(parser.parse_args())

    # read config
    with open("config/" + args["model"] + ".yml", "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    args.update(cfg)

    # dataset and dataloader
    test_dataset = CelebA_HQ_Dataset("data/test_data.txt", 64, 256, args)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

    model, _ = initialize_model_and_optimizer(args)
    model.load_state_dict(torch.load(args["checkpoint_path"]))
    model.eval()
    psnr = AverageMeter()
    with tqdm(total=len(test_dataset)-len(test_dataset)%4) as t:
        t.set_description(args['model'])
        for (LR_img, HR_img) in test_dataloader:
            LR_img, HR_img = LR_img.to(device()), HR_img.to(device())
            with torch.no_grad():
                SR_img = model(LR_img)
                SR_img = SR_img.clamp(0, 255)
            psnr.update(calculate_PSNR(SR_img, HR_img), len(LR_img))
            t.update(len(LR_img))

    print('Test PSNR: {:.2f}'.format(psnr.avg))

    # Sample
    # for (LR_img, HR_img) in test_dataloader:
    #     SR_img = model(LR_img)
    #     SR_img = transforms.ToPILImage()(SR_img[0])
    #     HR_img = transforms.ToPILImage()(HR_img[0])
    #     SR_img.save("output/test_SR.png")
    #     HR_img.save("output/test_GT.png")
    #     break
