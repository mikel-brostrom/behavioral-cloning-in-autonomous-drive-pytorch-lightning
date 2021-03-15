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


IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 16, 32, 1
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


class CarDataset(Dataset):
    """
    Dataset wrapping images and target steering_angle.
    """

    def __init__(self, X, y, is_training, transform=None):
        self.X = X
        self.y = y
        self.is_training = is_training
        self.transform = transform
        self.jpg_reader = TurboJPEG()

        self.transformations = transforms.Compose([transforms.Lambda(lambda x: x/127.5 - 1)])

    def __getitem__(self, index):
        image_path = self.X[index]
        steering_angle = self.y[index]
        
        # read image
        with open(image_path, "rb") as jpgfile:
                image = self.jpg_reader.decode(
                    jpgfile.read()
                )

        image = self.preprocess(image)
        image, steering_angle = self.augument_img(image, steering_angle)

        #print('\nBefore preprocess', image.shape)
        
        image = self.transformations(image)
        #image = image.reshape(image.shape[0], image.shape[-1], image.shape[1], image.shape[2])

        #print('\nAfter preprocess', image[0].shape)
        #print(type(steering_angle))

        return image, steering_angle
    
    def __len__(self):
        return self.X.shape[0]

    def augument_img(self, image, steering_angle):
        if np.random.rand() < 0.6:
            image, steering_angle = self.random_flip(image, steering_angle)
            image, steering_angle = self.random_translate(image, steering_angle, 100, 10)
            image = self.random_shadow(image)
            image = self.random_brightness(image)
        return image, steering_angle

    def crop(self, image):
        """
        Crop the image (removing the sky at the top and the car front at the bottom)
        """
        return image[60:-25, :, :] # remove the sky and the car front

    def rgb2hsv(self, image):
        """
        Convert the image from RGB to HSV 
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    def preprocess(self, image):
        """
        Combine all preprocess functions into one
        """
        image = self.crop(image)
        image = self.rgb2hsv(image)
        #image = self.resize(image)

        return image


    def resize(self, image):
        """
        Resize the image to the input shape used by the network model
        """
        return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


    def rgb2yuv(self, image):
        """
        Convert the image from RGB to YUV (This is what the NVIDIA model does)
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    

    def random_flip(self, image, steering_angle):
        """
        Randomly flipt the image left <-> right, and adjust the steering angle.
        """
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            steering_angle = -steering_angle
        return image, steering_angle

    def random_translate(self, image, steering_angle, range_x, range_y):
        """
        Randomly shift the image virtially and horizontally (translation).
        """
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)
        steering_angle += trans_x * 0.002
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))
        return image, steering_angle

    def random_shadow(self, image):
        """
        Generates and adds random shadow
        """
        # (x1, y1) and (x2, y2) forms a line
        # xm, ym gives all the locations of the image
        x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
        x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
        xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

        # mathematically speaking, we want to set 1 below the line and zero otherwise
        # Our coordinate is up side down.  So, the above the line: 
        # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
        # as x2 == x1 causes zero-division problem, we'll write it in the below form:
        # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
        mask = np.zeros_like(image[:, :, 1])
        mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

        # choose which side should have shadow and adjust saturation
        cond = mask == np.random.randint(2)
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        # adjust Saturation in HLS(Hue, Light, Saturation)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    def random_brightness(self, image):
        """
        Randomly adjust brightness of the image.
        """
        # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:,:,2] =  hsv[:,:,2] * ratio
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)