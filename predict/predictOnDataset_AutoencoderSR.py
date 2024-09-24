# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:12:03 2024

@author: MargaridaP
"""

# PyTorch
from torch import nn
from torch import cuda
from torch.utils.data import DataLoader
from torch import no_grad
from torch import load as torchload
from torch import stack as torchstack
from torch.utils.data.sampler import SubsetRandomSampler

import os
import csv
import time
import numpy as np
import torch

from matplotlib import pyplot as plt

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import sys

sys.path.insert(0, '..')
import utils.utils_dataAEct
from utils.utils_model import get_model, save_model_params, load_model_params
from utils.utils_transforms import CTTransform
from utils.utils_csv import readCsvDataLists
from utils.utils_train import dataPredictions
import utils.utils_paths


def main(model_folder, datasetlist, subset=None, dispFlag=False):
    # Load model params
    model_params = load_model_params(model_folder)

    nmodels = 0
    nfolds = model_params['nfolds']
    for k_ind in range(nfolds):
        if os.path.isfile(os.path.join(model_folder, 'model_{}.pt'.format(k_ind))):
            nmodels += 1
        else:
            break

    # Image transformations
    image_transform = CTTransform(model_params, train=False)

    if nmodels > len(datasetlist):
        for datastr in datasetlist:
            dataset, path_results = loadData(datastr, image_transform, model_folder, subset)
            for k_ind in range(nmodels):
                if os.path.isfile(os.path.join(path_results, 'fold{}predictions.csv'.format(k_ind))):
                    print('Predictions already made for fold {} - {}.'.format(k_ind, datastr))
                    continue

                if not os.path.isfile(os.path.join(model_folder, 'fold{}history.csv'.format(k_ind))):
                    print('Model fold {} has not finished training.'.format(k_ind))
                    continue

                model = loadModel(model_folder, k_ind, model_params)
                dataloader = getDataLoader(dataset, datastr, subset, k_ind, nfolds)
                #dataloader = getDataLoader(dataset, datastr, testfold, k_ind, nfolds)
                predict(model, dataloader, path_results, k_ind, dispFlag=dispFlag, dataset=dataset)
    else:
        for k_ind in range(0, nmodels + 1):
            if not os.path.isfile(os.path.join(model_folder, 'fold{}history.csv'.format(k_ind))):
                print('Model fold {} has not finished training.'.format(k_ind))
                continue

            model = loadModel(model_folder, k_ind, model_params)
            for datastr in datasetlist:
                dataset, path_results = loadData(datastr, image_transform, model_folder, subset)
                if os.path.isfile(os.path.join(path_results, 'fold{}predictions.csv'.format(k_ind))):
                    print('Predictions already made for fold {} - {}.'.format(k_ind, datastr))
                    continue
                dataloader = getDataLoader(dataset, datastr, subset, k_ind, nfolds)
                predict(model, dataloader, path_results, k_ind, dispFlag=dispFlag, dataset=dataset)


def loadData(datastr, image_transform, model_folder, subset):
    datafolder = os.path.split(os.path.split(datastr)[0])[-1]
    if subset is None:
        path_results = os.path.join(model_folder, datafolder)
    else:
        path_results = os.path.join(model_folder, datafolder + '_' + subset)

    if not os.path.isdir(path_results):
        os.mkdir(path_results)

    # Datasets from each folder
    dataset = utils.utils_dataAEct.DatasetCSV(root=datastr, transform=image_transform)
    print('Found {} samples.'.format(len(dataset.samples)))

    return dataset, path_results

# def loadModel(model_folder, k_ind, model_params):
#     model_path = os.path.join(model_folder, 'model_{}.pt'.format(k_ind))
#     print('Loading', model_path)
#     # Load model
    
#     model = get_model(model_params)
#     model_dict = torchload(model_path)
#     model_dict = {k: model_dict[k] for k in model.state_dict()}
#     model.load_state_dict(model_dict)
    
#     if next(model.parameters()).is_cuda:
#         model = model.module
        
#     return model

def loadModel(model_folder, k_ind, model_params):
    model_path = os.path.join(model_folder, 'model_{}.pt'.format(k_ind))
    print('Loading', model_path)
    
    model = get_model(model_params)
    model_dict = model.state_dict()
    pretrained_dict = torchload(model_path)
    
    # Filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    
    model_dict.update(pretrained_dict)
    
    # Load the new state dict
    model.load_state_dict(model_dict)
    
    if next(model.parameters()).is_cuda:
        model = model.module
        
    return model


def getDataLoader(dataset, datastr, subset, k_ind, nfolds):
    # Define data loading
    if subset in ['train', 'val', 'test']:
        test_k, val_k, train_k = (k_ind - 1) % nfolds, k_ind, [kk for kk in range(nfolds) if
                                                               not kk in [k_ind, (k_ind - 1) % nfolds]]
        data_k = {'test': test_k, 'val': val_k, 'train': [train_k]}
        data_index = dataset.get_sampler_index(data_k[subset])
    else:
        data_index = [i for i, img in enumerate(dataset.samples)]

    data_sampler = SubsetRandomSampler(data_index[0])
    print('{} set: {}'.format(datastr, len(data_index)))
    # Dataloader iterators
    dataloader = DataLoader(dataset, batch_size=10, sampler=data_sampler, shuffle=False)

    return dataloader

def weighted_average_interpolation(slice1, slice2, distance1, distance2):
    interpolated_slice = (distance2 * slice1 + distance1 * slice2) / (distance1 + distance2)
    return interpolated_slice

def predict(model, dataloader, path_results, k_ind, dataset, dispFlag=False):
    path_results_k = os.path.join(path_results, 'fold{}predictions'.format(k_ind))
    if not os.path.isdir(os.path.join(path_results_k)):
        os.mkdir(path_results_k)

    dataloader.dataset.return_paths = True
    dPredictions = dataPredictions()
    with no_grad():
        model.eval()
        nbatches = len(dataloader)
        st_time = time.time()
        for ii, (data, target, paths) in enumerate(dataloader):
            
            ct1, ct2 = torch.chunk(data, 2, dim=2)
   
            # ct1= torch.cat((ct1, ct1), dim=2)#para ter 2 canais
            # ct2= torch.cat((ct2, ct2), dim=2)
            
            ct1 = ct1.cuda()
            ct2 = ct2.cuda()

            encoder_output1 = []
            for module in model.down_blocks:
                ct1 = module(ct1)
                
            ct1_encoded = ct1
            
            encoder_output2 = []
            for module in model.down_blocks:
              ct2 = module(ct2)
              
            ct2_encoded = ct2
            
            pos_ct1 = dataset.get_samples_ctpos(ii)[0]
            pos_ct2 = dataset.get_samples_ctpos(ii)[1]
            pos_x = dataset.get_samples_ctcapos(ii)[1]#alterar aqui qual a posicao de ctca desjeada
            
            d1= abs(pos_x-pos_ct1)
            d2= abs(pos_x-pos_ct2)
            
            weighted_avg_output = weighted_average_interpolation(ct1_encoded, ct2_encoded, d1, d2)
            
            for i, module in enumerate(model.up_blocks):
                weighted_avg_output = module(weighted_avg_output)
                
            output = model.conv_final(weighted_avg_output)
            tanh_activation = torch.nn.Tanh()
            output = tanh_activation(output) * 1000.0 
            #output = model(data)

            if dispFlag:
                for d, o in zip(data, output):
                    image = d.cpu().permute(1, 2, 0).detach().numpy()
                    segm = o.cpu().permute(1, 2, 0).detach().numpy()
                    fig, (ax0, ax1) = plt.subplots(1, 2)
                    ax0.imshow(image)
                    ax1.imshow(segm)
                    plt.show()

            dPredictions.append(data, output, target=target, fnames=paths)
            print(
                f'\rFold {k_ind}: \t{100 * (ii + 1) / nbatches:.2f}% complete. {time.time() - st_time:.2f} seconds elapsed in fold.',
                end='')
            dPredictions.write(path_results_k, clear=True)

    print(
        f'Fold {k_ind}: \t{100 * (ii + 1) / nbatches:.2f}% complete. {time.time() - st_time:.2f} seconds elapsed in fold.')


if __name__ == '__main__':
    model_folder = os.path.join(utils.utils_paths.mainpath, 'results', 'AutoencoderCT_LR1e-4')
    datasetlist = [os.path.join(utils.utils_paths.datapaths['cardiac_ct'], 'cardiac_ct_ctca1.csv')]
    #datasetlist = [os.path.join(utils.utils_paths.datapaths['cardiac_fat'], 'cardiac_ct_unet.csv')]
    main(model_folder, datasetlist, subset='test', dispFlag=False)
