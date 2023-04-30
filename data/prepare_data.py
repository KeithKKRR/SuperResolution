import os

import cv2

CelebA_HQ_root = "D:/CelebAMask-HQ/CelebA-HQ-img"
target_root_64 = "D:/SuperResolution/data/CelebA-HQ-64"
target_root_128 = "D:/SuperResolution/data/CelebA-HQ-128"
target_root_256 = "D:/SuperResolution/data/CelebA-HQ-256"
target_root_512 = "D:/SuperResolution/data/CelebA-HQ-512"

cnt = 0
for img_name in os.listdir(CelebA_HQ_root):
    print(cnt)
    cnt += 1
    original_img = cv2.imread(os.path.join(CelebA_HQ_root, img_name))
    img64 = cv2.resize(original_img, (64, 64))
    cv2.imwrite(os.path.join(target_root_64, img_name), img64)
    # img128 = cv2.resize(original_img, (128, 128))
    # cv2.imwrite(os.path.join(target_root_128, img_name), img128)
    img256 = cv2.resize(original_img, (256, 256))
    cv2.imwrite(os.path.join(target_root_256, img_name), img256)
    # img512 = cv2.resize(original_img, (512, 512))
    # cv2.imwrite(os.path.join(target_root_512, img_name), img512)

