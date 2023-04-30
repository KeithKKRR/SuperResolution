import os
import random

import cv2

CelebA_HQ_root = "D:/CelebAMask-HQ/CelebA-HQ-img"
target_root_64 = "D:/SuperResolution/data/CelebA-HQ-64"
target_root_128 = "D:/SuperResolution/data/CelebA-HQ-128"
target_root_256 = "D:/SuperResolution/data/CelebA-HQ-256"
target_root_512 = "D:/SuperResolution/data/CelebA-HQ-512"


# Generate Images in Different Resolution
def generate_image_in_different_resolution():
    cnt = 0
    for img_name in os.listdir(CelebA_HQ_root):
        print(cnt)
        cnt += 1
        original_img = cv2.imread(os.path.join(CelebA_HQ_root, img_name))
        # Generate 64x64 Images
        img64 = cv2.resize(original_img, (64, 64))
        cv2.imwrite(os.path.join(target_root_64, img_name), img64)
        # Generate 128x128 Images
        img128 = cv2.resize(original_img, (128, 128))
        cv2.imwrite(os.path.join(target_root_128, img_name), img128)
        # Generate 256x256 Images
        img256 = cv2.resize(original_img, (256, 256))
        cv2.imwrite(os.path.join(target_root_256, img_name), img256)
        # Generate 512x512 Images
        img512 = cv2.resize(original_img, (512, 512))
        cv2.imwrite(os.path.join(target_root_512, img_name), img512)


# Split Data into Train, Validation, and Test
def split_data():
    train_proportion = 0.8
    val_proportion = 0.1
    test_proportion = 0.1
    data = list(range(30000))
    random.shuffle(data)
    with open("data/train_data.txt", "w", encoding="utf-8") as f:
        for index in range(0, int(30000 * train_proportion)):
            f.write(str(data[index]) + ".jpg\n")

    with open("data/val_data.txt", "w", encoding="utf-8") as f:
        for index in range(int(30000 * train_proportion), int(30000 * train_proportion + 30000 * val_proportion)):
            f.write(str(data[index]) + ".jpg\n")

    with open("data/test_data.txt", "w", encoding="utf-8") as f:
        for index in range(int(30000 * train_proportion + 30000 * val_proportion), 30000):
            f.write(str(data[index]) + ".jpg\n")


if __name__ == '__main__':
    generate_image_in_different_resolution()
    split_data()
