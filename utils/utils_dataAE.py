import torch.utils.data as data
import torch

from PIL import Image
from typing import Dict
import numpy as np
from matplotlib import pyplot as plt
from torchvision import get_image_backend
from torch.utils.data.sampler import Sampler
import os
import os.path
import csv
import random
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

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


def default_loader(path):
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    elif os.path.splitext(path)[-1] == '.npy':
        return npy_loader(path)
    else:
        return pil_loader(path)

class DatasetCSV(data.Dataset):
    def __init__(self, root, loader=default_loader, extensions=IMG_EXTENSIONS, transform=None, return_paths=False):
        if not isinstance(root, list):
            root = [root]

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.transform = transform

        self.make_dataset(self.root, reset_samples=True)
        self.return_paths = return_paths
        
    def __getitem__(self, index):   
        path_ct, path_ctca = self.samples[index]
        ct = self.loader(path_ct.replace('\\', os.sep))
        ctca = self.loader(path_ctca.replace('\\', os.sep))

        if self.transform is not None:
            ct, ctca = self.transform(ct, ctca)
        
        ct = np.expand_dims(ct, axis=0)    
        ctca = np.expand_dims(ctca, axis=0)    
        
        if self.return_paths:
            return ct, ctca
        else:
            return ct, ctca
    
    # def __getitem__(self, index):
    #     if isinstance(index, int):
    #         path_ct, path_ctca = self.samples[index]
    #         ct = self.loader(path_ct.replace('\\', os.sep))
    #         ctca = self.loader(path_ctca.replace('\\', os.sep))
    
    #         if self.transform is not None:
    #             ct, ctca = self.transform(ct, ctca)
    
    #         ct = np.expand_dims(ct, axis=0)    
    #         ctca = np.expand_dims(ctca, axis=0)    
    
    #         if self.return_paths:
    #             return ct, ctca
    #         else:
    #             return ct, ctca
    
    #     elif isinstance(index, tuple):
    #         path_ct, path_ctca = self.samples[index[0]]
    #         path_ct_next, path_ctca_next = self.samples[index[1]]
    
    #         ct = self.loader(path_ct.replace('\\', os.sep))
    #         ctca = self.loader(path_ctca.replace('\\', os.sep))
    #         ct_next = self.loader(path_ct_next.replace('\\', os.sep))
    #         ctca_next = self.loader(path_ctca_next.replace('\\', os.sep))
    
    #         if self.transform is not None:
    #             ct, ctca = self.transform(ct, ctca)
    #             ct_next, ctca_next = self.transform(ct_next, ctca_next)
    
    #         ct = np.expand_dims(ct, axis=0)    
    #         ctca = np.expand_dims(ctca, axis=0)
    #         ct_next = np.expand_dims(ct_next, axis=0)    
    #         ctca_next = np.expand_dims(ctca_next, axis=0)
    
    #         ct_batch = np.concatenate([ct, ct_next], axis=0)
    #         ctca_batch = np.concatenate([ctca, ctca_next], axis=0)
    
    #         if self.return_paths:
    #             return ct_batch, ctca_batch
    #         else:
    #             return ct_batch, ctca_batch
  
       
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
            self.ctca_slices = []
            self.sample_folds = []

        for rootdir in rootdirlist:
            rootdir = os.path.expanduser(rootdir)

            gt_dlist = readCsvDataLists(rootdir, targetstr=['CT', 'CTCA', 'Slices CTCA', 'Fold'])
            gt_ctlist = gt_dlist[0]
            gt_ctcalist = gt_dlist[1]
            gt_sliceslist = gt_dlist[2]
            gt_fldlist = gt_dlist[3]
            if len(gt_ctlist) != len(np.unique(gt_ctlist)):
                seen = []
                print('Duplicates:')
                for ct in gt_ctlist:
                    if ct in seen:
                        print(ct)
                    else:
                        seen.append(ct)
                raise (RuntimeError(
                    "There are {} repeated files on GT!!!".format(len(gt_ctlist) - len(np.unique(gt_ctlist)))))

            for ind, (ct, ctca, num_slices_ctca, fold) in enumerate(zip(gt_ctlist, gt_ctcalist, gt_sliceslist, gt_fldlist)):
                if has_file_allowed_extension(ct, self.extensions) and has_file_allowed_extension(ctca, self.extensions):
                    path_ct = os.path.join(os.path.split(rootdir)[0], ct)
                    path_ctca = os.path.join(os.path.split(rootdir)[0], ctca)
                    self.samples.append((path_ct, path_ctca))
                    self.ctca_slices.append(int(num_slices_ctca))
                    self.sample_folds.append(int(fold))
                else:
                    print('Did not recognize extension of file {}.'.format(ct))

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

   
# class CustomBatchSampler(Sampler):
#     def __init__(self, index_list, slice_counts, dataset, batch_size):
#         self.index_list = index_list
#         self.slice_counts = slice_counts
#         self.dataset = dataset
#         self.batch_size = batch_size

#     def __iter__(self):
#         while True:
#             batch = []
            
#             first_idx = random.choice(self.index_list)
#             num_slices = self.dataset.get_num_slices(first_idx)
#             next_idx_candidates = self.slice_counts.get(num_slices, [])
#             while not next_idx_candidates:
#                     first_idx = random.choice(self.index_list)
#                     num_slices = self.dataset.get_num_slices(first_idx)
#                     next_idx_candidates = self.slice_counts.get(num_slices, [])
#             next_idx = random.choice(next_idx_candidates)
#             while next_idx == first_idx or next_idx not in self.index_list:
#                     next_idx = random.choice(next_idx_candidates)
#             batch.append((first_idx, next_idx))
         
#             yield batch

    # def __len__(self):
    #     return len(self.index_list) // self.batch_size

# class CustomBatchSampler(Sampler):
#     def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool, data):
#         self.sampler = sampler
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         self.data = data

#     # def __iter__(self) -> Iterator[List[int]]:
#     #     if self.drop_last:
#     #         sampler_iter = iter(self.sampler)
#     #         while True:
#     #             batch = []
#     #             slice_count = None
#     #             for _ in range(self.batch_size):
#     #                 idx = next(sampler_iter)
#     #                 ctca_slice_count = self.data.ctca_slices[idx]
                    
#     #                 while slice_count is not None and ctca_slice_count != slice_count:
#     #                     idx = next(sampler_iter)
#     #                     ctca_slice_count = self.data.ctca_slices[idx]
#     #                 batch.append(idx)
#     #                 slice_count = ctca_slice_count
#     #             yield batch
    
#     def __iter__(self) -> Iterator[List[int]]:
#             if self.drop_last:
#                 sampler_iter = iter(self.sampler)
#                 while True:
#                     batch = []
#                     slice_count = None
#                     while len(batch) < self.batch_size:
#                         try:
#                             idx = next(sampler_iter)
#                         except StopIteration:
#                             break
#                         ctca_slice_count = self.data.ctca_slices[idx]
#                         if slice_count is None:
#                             slice_count = ctca_slice_count
#                         elif ctca_slice_count!= slice_count:
#                             continue
#                         batch.append(idx)
#                     if len(batch) == self.batch_size:
#                         yield batch
#                     else:
#                         break 

#             else:
#                 sampler_iter = iter(self.sampler) 
#                 batch = []
#                 slice_count = None
#                 for _ in range(len(self.sampler)):  
#                     idx = next(sampler_iter)  
#                     ctca_slice_count = self.data.ctca_slices[idx]
#                     while slice_count is not None and ctca_slice_count != slice_count:
#                         idx = next(sampler_iter)
#                         ctca_slice_count = self.data.ctca_slices[idx]
#                     batch.append(idx)
#                     slice_count = ctca_slice_count
#                     if len(batch) == self.batch_size:
#                         yield batch
#                         batch = []
#                 if batch:
#                     yield batch
            
#     def __len__(self) -> int:
#         if self.drop_last:
#             return len(self.sampler) // self.batch_size
#         else:
#             return (len(self.sampler) + self.batch_size - 1) // self.batch_size

#     def __getitem__(self, index):
#         sampler_iter = iter(self.sampler)
#         for _ in range(index + 1):
#             next(sampler_iter)
#         return next(sampler_iter)


class CustomBatchSampler(Sampler):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool, data):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data = data

    def __iter__(self) -> Iterator[List[int]]:
        sampler_iter = iter(self.sampler)
        
        while True:
            batch = []
            try:
                idx = next(sampler_iter)
            except StopIteration:
                break
            slice_count = self.data.ctca_slices[idx]
            batch.append(idx)
            while len(batch) < self.batch_size:
                next_idx = self.find_next_index_with_slice_count(slice_count)
                if next_idx is not None:
                    batch.append(next_idx)
                    if len(batch) == self.batch_size:
                        yield batch
                        break
                else:
                    break
            else:
                yield batch

    def find_next_index_with_slice_count(self, slice_count):
        indices_with_same_slice_count = self.data.ctca_slices_index[slice_count] 
        idx = np.random.choice(indices_with_same_slice_count)
        return idx

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        sampler_iter = iter(self.sampler)
        for _ in range(index + 1):
            next(sampler_iter)
        return next(sampler_iter)

