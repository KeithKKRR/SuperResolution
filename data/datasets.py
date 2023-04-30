import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CelebA_HQ_Dataset(Dataset):
    def __init__(self, data_file, LR_res, HR_res):
        super(CelebA_HQ_Dataset, self).__init__()
        self.LR_res = LR_res
        self.HR_res = HR_res
        self.transform = transforms.ToTensor()
        with open(data_file, "r", encoding="utf-8") as f:
            self.img_names = f.readlines()
        for i in range(len(self.img_names)):
            self.img_names[i] = self.img_names[i].rstrip("\n")

    def __getitem__(self, idx):
        LR_img_path = os.path.join("data", "CelebA-HQ-" + str(self.LR_res), self.img_names[idx])
        LR_img_tensor = self.transform(Image.open(LR_img_path).convert("RGB"))
        HR_img_path = os.path.join("data", "CelebA-HQ-" + str(self.HR_res), self.img_names[idx])
        HR_img_tensor = self.transform(Image.open(HR_img_path).convert("RGB"))
        return LR_img_tensor, HR_img_tensor

    def __len__(self):
        return len(self.img_names)


# test code
# train_dataset = CelebA_HQ_Dataset("data/train_data.txt", 64, 256)
# train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=False)
# for index, (LR_img, HR_img) in enumerate(train_dataloader):
#     print(LR_img.shape, HR_img.shape)
#     break
