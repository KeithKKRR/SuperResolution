import os

import cv2

CelebA_HQ_root = "D:/CelebAMask-HQ/CelebA-HQ-img"
LR_target_root = "D:/SuperResolution/data/CelebA-HQ-128"
HR_target_root = "D:/SuperResolution/data/CelebA-HQ-512"

cnt = 0
for img_name in os.listdir(CelebA_HQ_root):
    print(cnt)
    cnt += 1
    original_img = cv2.imread(os.path.join(CelebA_HQ_root, img_name))
    LR_img = cv2.resize(original_img, (128, 128))
    cv2.imwrite(os.path.join(LR_target_root, img_name), LR_img)
    HR_img = cv2.resize(original_img, (512, 512))
    cv2.imwrite(os.path.join(HR_target_root, img_name), HR_img)

