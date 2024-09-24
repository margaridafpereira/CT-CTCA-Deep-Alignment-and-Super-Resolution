try:
    from torchvision import models
    import torch.nn as nn
    import torch
    import torch.nn.functional as F
except:
    print('Could not load PyTorch... Continuing anyway!')

import os
import numpy as np
import pickle
from models.autoencoder_ctca import Autoencoder_CTCA 
from models.autoencoder_ct import Autoencoder_CT 
from models.autoencoder_ct_ctca import Autoencoder_CT_CTCA 
from models.autoencoder2E2D import Autoencoder_2E2D 
from models.AE2D import Autoencoder_AE2D 
from models.unet import UNet
from models.unetSR import UNetSR
from torch.nn.parallel import DistributedDataParallel as DDP

import torch
import torch.nn as nn
#import torchxrayvision as xrv

def get_model(model_params):
    model = get_pretrained_model(model_params)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[0,1])
    else:
        model = nn.DataParallel(model)

    model.to('cuda')

    return model

# def get_model(model_params):
#     model = get_pretrained_model(model_params)

#     if torch.cuda.is_available():
#         device = torch.device("cuda:1")  
#         model = nn.DataParallel(model, device_ids=[1])  
#         model.to(device)  
#         print("Modelo criado e movido para a GPU 1.")
   

#     return model

def load_model_params(fpath):
    model_params = pickle.load(open(os.path.join(fpath, 'modelparams.pkl'), "rb"))
    return model_params


def save_model_params(fpath, model_params):
    with open(os.path.join(fpath, 'modelparams.pkl'), 'wb') as f:
        pickle.dump(model_params, f)


def get_pretrained_model(model_params):

    if model_params['model_name'] == 'autoencoder_ct_ctca':
        if not 'in_channels' in model_params:
            model_params['in_channels'] = 1
        if not 'out_channels' in model_params:
            model_params['out_channels'] = 1
        autoencoderct_ctca_params = {'in_channels': 1, 'out_channels': model_params['out_channels'],
                        'n_blocks': 5, 'start_filters': 32, 'activation': 'relu', 'normalization': 'batch',
                        'conv_mode': 'same', 'dim': 2, 'up_mode': 'transposed'}
        for k, _ in autoencoderct_ctca_params.items():
            if k in model_params:
                autoencoderct_ctca_params[k] = model_params[k]

        model = Autoencoder_CT_CTCA(**autoencoderct_ctca_params)
  
    
    elif model_params['model_name'] == 'autoencoder2D':
        if not 'in_channels' in model_params:
            model_params['in_channels'] = 1
        if not 'out_channels' in model_params:
            model_params['out_channels'] = 1
        autoencoderAE2D_params = {'in_channels': 1, 'out_channels': model_params['out_channels'],
                        'n_blocks': 5, 'start_filters': 32, 'activation': 'relu', 'normalization': 'batch',
                        'conv_mode': 'same', 'dim': 2, 'up_mode': 'transposed'}
        for k, _ in autoencoderAE2D_params.items():
            if k in model_params:
                autoencoderAE2D_params[k] = model_params[k]

        model = Autoencoder_AE2D(**autoencoderAE2D_params)
        
    # elif model_params['model_name'] == 'autoencoder_2E2D':
    #     if not 'in_channels' in model_params:
    #         model_params['in_channels'] = 1
    #     if not 'out_channels' in model_params:
    #         model_params['out_channels'] = 1
    #     autoencoder_2E2D_params = {'in_channels': model_params['in_channels'], 'out_channels': model_params['out_channels'],
    #                     'n_blocks': 3, 'start_filters': 32, 'activation': 'relu', 'normalization': 'batch',
    #                     'conv_mode': 'same', 'dim': 3, 'up_mode': 'transposed'}
    #     for k, _ in autoencoder_2E2D_params.items():
    #         if k in model_params:
    #             autoencoder_2E2D_params[k] = model_params[k]
    
    #     model = Autoencoder_2E2D(**autoencoder_2E2D_params)

        
    elif model_params['model_name'] == 'autoencoder_ct':
        if not 'in_channels' in model_params:
            model_params['in_channels'] = 1
        if not 'out_channels' in model_params:
            model_params['out_channels'] = 1
        autoencoderct_params = {'in_channels': model_params['in_channels'], 'out_channels': model_params['out_channels'],
                        'n_blocks': 3, 'start_filters': 32, 'activation': 'relu', 'normalization': 'batch',
                        'conv_mode': 'same', 'dim': 3, 'up_mode': 'transposed'}
        for k, _ in autoencoderct_params.items():
            if k in model_params:
                autoencoderct_params[k] = model_params[k]

        model = Autoencoder_CT(**autoencoderct_params)
            
    # elif model_params['model_name'] == 'autoencoder_ctca':
    #     if not 'in_channels' in model_params:
    #         model_params['in_channels'] = 1
    #     if not 'out_channels' in model_params:
    #         model_params['out_channels'] = 1
    #     autoencoderctca_params = {'in_channels': model_params['in_channels'], 'out_channels': model_params['out_channels'],
    #                     'n_blocks': 3, 'start_filters': 32, 'activation': 'relu', 'normalization': 'batch',
    #                     'conv_mode': 'same', 'dim': 3, 'up_mode': 'transposed'}
    #     for k, _ in autoencoderctca_params.items():
    #         if k in model_params:
    #             autoencoderctca_params[k] = model_params[k]

    #     model = Autoencoder_CTCA(**autoencoderctca_params)
    
    elif model_params['model_name'] == 'unet_weighted_average':
        if not 'in_channels' in model_params:
            model_params['in_channels'] = 1
        if not 'out_channels' in model_params:
            model_params['out_channels'] = 1
        unet_params = {'in_channels': model_params['in_channels'], 'out_channels': model_params['out_channels'],
                        'n_blocks': 5, 'start_filters': 32, 'activation': 'relu', 'normalization': 'batch',
                        'conv_mode': 'same', 'dim': 2, 'up_mode': 'transposed'}
        for k, _ in unet_params.items():
            if k in model_params:
                unet_params[k] = model_params[k]

        model = UNetSR(**unet_params)
        
    elif model_params['model_name'] == 'unet':
        if not 'in_channels' in model_params:
            model_params['in_channels'] = 1
        if not 'out_channels' in model_params:
            model_params['out_channels'] = 1
        unet_params = {'in_channels': 1, 'out_channels': model_params['out_channels'],
                        'n_blocks': 5, 'start_filters': 32, 'activation': 'relu', 'normalization': 'batch',
                        'conv_mode': 'same', 'dim': 2, 'up_mode': 'transposed'}
        for k, _ in unet_params.items():
            if k in model_params:
                unet_params[k] = model_params[k]

        model = UNet(**unet_params)
    else:
        print('No model', model_params['model_name'])
        assert True == False

    return model
