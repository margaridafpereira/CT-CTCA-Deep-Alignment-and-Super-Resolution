import os
import numpy as np
import torch
from scipy.ndimage import rotate
import cv2

dispFlag = True
if dispFlag:
    from matplotlib import pyplot as plt

class CTTransform(object):
    def __init__(self, model_params, toTensor=True, train=True):
        self.range = model_params['augm_params']['img_range']
        self.crop = model_params['augm_params']['crop']
        self.random_crop = RandomCrop(width=self.crop, height=self.crop)
        self.rotate = model_params['augm_params']['rotate']
        self.random_rotation = RandomRotation(degrees=self.rotate, fill=model_params['augm_params']['img_range'][0])
        self.random_flip = RandomFlip(flip_type=model_params['augm_params']['flip'])
        self.size_input = model_params['size_input']
        self.resize = Resize(self.size_input)
        self.toTensor = toTensor
        self.train = train
        
    def __call__(self, img, tgt):
        #Merge img and tgt
        n_layers = [img.shape[0], tgt.shape[0]]
        
        #img = np.concatenate((img, tgt))
        
        img[np.isnan(img)] = -1000
        tgt[np.isnan(tgt)] = -1000
        
        # Clip HU values
        img[img < self.range[0]] = self.range[0]
        img[img > self.range[1]] = self.range[1]
        
        tgt[tgt < self.range[0]] = self.range[0]
        tgt[tgt > self.range[1]] = self.range[1] 

        if self.train:
            #Apply augmentations
            img = self.random_crop(img)
            img = self.random_rotation(img)
            img = self.random_flip(img)
            
            tgt = self.random_crop(tgt)
            tgt = self.random_rotation(tgt)
            tgt = self.random_flip(tgt)

        #Resize
        img = self.resize(img)
        tgt = self.resize(tgt)

        # Split img and tgt
        img = img.astype(np.float32)
        tgt = tgt.astype(np.float32)
        
        # tgt = img[n_layers[0]:, :, :]
        # img = img[:n_layers[0], :, :]

        # Convert to Tensor
        if self.toTensor:
            img = torch.from_numpy(img)
            tgt = torch.from_numpy(tgt)

        return img, tgt

class Resize(object):
    def __init__(self, size=512):
        self.size = size

    def __call__(self, img):
        imgsize = list(img.shape)
        imgsize[-1] = self.size
        imgsize[-2] = self.size
        nimg = np.zeros(imgsize)

        for i, ig in enumerate(img):
            nimg[i, :, :] = cv2.resize(ig, dsize=[self.size, self.size])

        return nimg


# class Resize(object): #para CTCA
#     def __init__(self, size=512):
#         self.size = size

#     def __call__(self, img):
#         n_channels, img_height, img_width = img.shape  
#         nimg = np.zeros((n_channels, self.size, self.size))  

#         for i in range(n_channels):
#             nimg[i, :, :] = cv2.resize(img[i, :, :], dsize=(self.size, self.size))

#         return nimg


class RandomCrop(object):
    def __init__(self, width=.11, height=.11):
        self.width = width
        self.height = height

    def __call__(self, img):
        w, h = self.get_params(img)
        new_img = self.apply_params(img, w, h)
        return new_img

    def get_params(self, img):
        w = int(self.width * np.random.uniform() * img.shape[-2] / 2)
        h = int(self.height * np.random.uniform() * img.shape[-1] / 2)
        return w, h

    def apply_params(self, img, w, h):
        imgsize = img.shape
        img = img[:, w:imgsize[-2]-w, h:imgsize[-1]-h]

        return img

class RandomRotation(object):
    def __init__(self, degrees=0, fill=0):
        self.degrees = degrees
        self.fill = fill

    def __call__(self, img):
        d = int(self.degrees * np.random.uniform() - self.degrees / 2)

        nimg = np.zeros(img.shape)
        for i, ig in enumerate(img):
            nimg[i, :, :] = rotate(ig, angle=d, reshape=False, cval=self.fill)
        return nimg   
    
class RandomFlip(object):
    def __init__(self, flip_type):
        self.flip_type = flip_type
        
    def __call__(self, img):
        if self.flip_type == 0:  
            img = np.flip(img, axis=2)
        elif self.flip_type == 1:  
            img = np.flip(img, axis=1)

        return img



