# -*- coding: utf-8 -*-
"""
Created on Mon May 20 00:00:38 2024

@author: MargaridaP
"""

import torch.utils.data as data
import torch

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from torchvision import get_image_backend

import os
import os.path
import csv

from utils.utils_csv import readCsvDataLists, readGTCsv, readPredCsv

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.npy']

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def npy_loader(path):
    img = np.load(path)
    if len(img.shape) < 3:
        img = img[None, :, :]
    return img


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        try:
            if np.max(img)>255:
                imgdata = np.asarray(img)
                imgdata = (imgdata+1) / 256 - 1
                img = Image.fromarray(np.uint8(imgdata))
        except:
            print(path)
            
        return img.convert('L')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def txt_loader(path):
    with open(path, 'r') as file:
        values = [float(line.strip()) for line in file.readlines()]
        return values 

def txt_loader1(path):
    with open(path, 'r') as file:
        return [float(pos) for line in file.readlines() for pos in line.strip()[1:-1].split(',')]


def default_loader(path):
    # if get_image_backend() == 'accimage':
    #     return accimage_loader(path)
    if os.path.splitext(path)[-1] == '.npy':
        return npy_loader(path)
    elif os.path.splitext(path)[-1] == '.txt':
        return txt_loader(path)
    else:
        return pil_loader(path)

class DatasetCSV(data.Dataset):
    def __init__(self, root, loader=default_loader, txt_loader1= txt_loader1,  extensions=IMG_EXTENSIONS, transform=None, return_paths=False):
        if not isinstance(root, list):
            root = [root]

        self.root = root
        self.loader = loader
        self.txt_loader1 = txt_loader1

        self.extensions = extensions

        self.transform = transform

        self.make_dataset(self.root, reset_samples=True)

        self.return_paths = return_paths

    def __getitem__(self, index):
        path, path_target = self.samples[index]
        sample = self.loader(path.replace('\\', os.sep))
        target = sample

        if self.transform is not None:
            sample, target = self.transform(sample, target)
        
        sample = np.expand_dims(sample, axis=0)    
        target = np.expand_dims(target, axis=0)
        
        if self.return_paths:
            return sample, target, path
        else:
            return sample, target


    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

        
    def make_dataset(self, rootdirlist, reset_samples=False):
        if reset_samples:
            self.samples = []
            self.sample_pos_ct = []
            self.sample_pos_ctca = []
            self.sample_folds = []

        for rootdir in rootdirlist:
            rootdir = os.path.expanduser(rootdir)

            gt_dlist = readCsvDataLists(rootdir, targetstr=['CT', 'CTCA', 'Slices CTCA',  'CT_pos', 'CTCA_pos', 'Fold'] )
            gt_imglist = gt_dlist[0]
            gt_tgtlist = gt_dlist[0]
            gt_posctlist = gt_dlist[3]
            gt_posctcalist = gt_dlist[4]
            gt_fldlist = gt_dlist[5]
            if len(gt_imglist) != len(np.unique(gt_imglist)):
                seen = []
                print('Duplicates:')
                for img in gt_imglist:
                    if img in seen:
                        print(img)
                    else:
                        seen.append(img)
                raise (RuntimeError(
                    "There are {} repeated files on GT!!!".format(len(gt_imglist) - len(np.unique(gt_imglist)))))

            for ind, (img, tgt, pos_ct, pos_ctca, fold) in enumerate(zip(gt_imglist, gt_tgtlist, gt_posctlist, gt_posctcalist, gt_fldlist)):
                if has_file_allowed_extension(img, self.extensions) and has_file_allowed_extension(tgt, self.extensions):
                    path = os.path.join(os.path.split(rootdir)[0], img)
                    path_tgt = os.path.join(os.path.split(rootdir)[0], tgt)
                    path_sample_pos_ct = os.path.join(os.path.split(rootdir)[0], pos_ct)
                    path_sample_pos_ctca = os.path.join(os.path.split(rootdir)[0], pos_ctca)
                    self.samples.append((path, path_tgt))
                    self.sample_pos_ct.append((path_sample_pos_ct))
                    self.sample_pos_ctca.append((path_sample_pos_ctca))
                    self.sample_folds.append(int(fold))
                else:
                    print('Did not recognize extension of file {}.'.format(img))

        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files!!!"))

    def sample_samples(self, index):
        self.samples = [s for i, s in enumerate(self.samples) if i in index]
        self.imgs = [s for i, s in enumerate(self.imgs) if i in index]


    def get_sampler_index(self, k_list, exclude_if_no_folds=True):
        if not isinstance(k_list, list):
            k_list = [[k_list]]
        else:
            k_list = [[k] if not isinstance(k, list) else k for k in k_list]

        sample_folds = np.array(self.sample_folds)
        sample_ind = np.arange(len(self.sample_folds))
        k_index = [[] for _ in k_list]
        for ind_split, k_ind in enumerate(k_list):
            for kk in k_ind:
                k_index[ind_split].extend(sample_ind[sample_folds == kk])

        return k_index

    
    def get_samples_ctpos(self, index):
        path_sample_pos_ct = self.sample_pos_ct[index]
        sample_pos_ct = self.loader(path_sample_pos_ct.replace('\\', os.sep))
      
        return sample_pos_ct
    
    def get_samples_ctcapos(self, index):
        path_sample_pos_ctca = self.sample_pos_ctca[index]
        sample_pos_ctca = self.txt_loader1(path_sample_pos_ctca.replace('\\', os.sep))
        
        return sample_pos_ctca
