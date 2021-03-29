import os
import json
from pathlib import Path

import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from turbojpeg import TurboJPEG
import torchvision.transforms as transforms
import albumentations as albu
from PIL import Image
from time import sleep
Image.MAX_IMAGE_PIXELS = 2


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 16, 32, 1
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


class CarDataset(Dataset):
    """
    Dataset wrapping images and target steering_angle.
    """

    def __init__(self, X, y, is_training, transform=None, debug=False):
        self.X = X
        self.y = y
        self.is_training = is_training
        self.transform = transform
        self.jpg_reader = TurboJPEG()
        self.debug = debug


    def __getitem__(self, index):
        image_path = self.X[index]
        steering_angle = self.y[index]
        
        # read image
        with open(image_path, "rb") as jpgfile:
                image = self.jpg_reader.decode(
                    jpgfile.read()
                )

        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image, steering_angle = random_flip(image, steering_angle)

        if self.is_training is True:
            self.image_transforms = albu.Compose(
                [
                    # (160, 320, 3) ---> (60-135, 320, 3), remove sky and car front
                    albu.Crop(0, 60, 320, 160-25),
                    albu.Blur(blur_limit=3, p=0.1),
                    albu.RGBShift(p=0.1),
                    albu.HueSaturationValue(p=0.1),
                    albu.RandomBrightnessContrast(p=0.1),
                    albu.RandomGamma(p=0.1),
                    albu.RandomShadow(p=0.1),
                    albu.Normalize(always_apply=True)
                ]
            )
        else:
            self.image_transforms = albu.Compose(
                [
                    albu.Crop(0, 60, 320, 160-25),
                    albu.Normalize(always_apply=True)
                ]
            )

        image = self.image_transforms(image=image)["image"]

        # visualize augmentations
        if self.debug is True:
            img = Image.fromarray(image, 'RGB')
            img.show()
            sleep(3)

        return image, steering_angle
    
    def __len__(self):
        return self.X.shape[0]

def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle