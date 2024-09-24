# PyTorch
import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch import load as torchload
from torch.utils.data.sampler import SubsetRandomSampler

import os
import csv
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import utils.utils_dataAE
from utils.utils_model import get_model, save_model_params, load_model_params
import utils.utils_trainAE
from utils.utils_transforms import CTTransform
import utils.utils_paths

def main(model_params, path_results):
    traincsvfpaths = model_params['train_paths']

    # Results paths
    if not os.path.exists(path_results):
        os.mkdir(path_results)
    path_results = os.path.join(path_results, model_params['model_folder'])
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    # Save model parameters to new results folder
    save_model_params(path_results, model_params)

    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()  
    print(f'Train on gpu: {train_on_gpu}')

    # Number of gpus
    multi_gpu = False
    if train_on_gpu:
        gpu_count = cuda.device_count()
        print(f'{gpu_count} gpus detected.')
        if gpu_count > 1:
            multi_gpu = True

    # Image transformations
    image_transform = CTTransform(model_params)

    # Datasets from each folder
    data = utils.utils_dataAE.DatasetCSV(root=traincsvfpaths, transform=image_transform)    
    print('Found {} samples for training.'.format(len(data.samples)))
    
    nfolds = model_params['nfolds']
    for k_ind in range(nfolds):
    #for k_ind in [2]: # Train single fold
        save_file_name = os.path.join(path_results, 'model_{}.pt'.format(k_ind))

        # Define data loading
        test_k, val_k, train_k = (k_ind - 1) % nfolds, k_ind, [kk for kk in range(nfolds) if not kk in [k_ind, (k_ind - 1) % nfolds]]
        k_index = data.get_sampler_index([val_k, train_k], exclude_if_no_folds=False)
        val_index, train_index = k_index[0], k_index[1]
        
        train_sampler = SubsetRandomSampler(train_index)
        val_sampler = SubsetRandomSampler(val_index)
        
        # dataloaders = {
        #     'train': DataLoader(data, batch_size=model_params['batch_size'], sampler=train_sampler, shuffle=False),
        #     'val':  DataLoader(data, batch_size=model_params['batch_size'],  sampler=val_sampler, shuffle=False)}
                
        # train_sampler = utils.utils_dataAE.CustomBatchSampler(train_index,  data.slice_counts, data, batch_size=model_params['batch_size'])
        # val_sampler = utils.utils_dataAE.CustomBatchSampler(val_index,  data.slice_counts, data, batch_size=model_params['batch_size'])
   
        batch_size=model_params['batch_size']
        drop_last=True
        
        batch_sampler_train = utils.utils_dataAE.CustomBatchSampler(train_sampler, batch_size, drop_last, data)
        batch_sampler_val = utils.utils_dataAE.CustomBatchSampler(val_sampler, batch_size, drop_last, data)
        
        print('Training set: {}'.format(len(train_index)))
        print('Validation set: {}'.format(len(val_index)))

        # Dataloader iterators
        dataloaders = {
            'train': DataLoader(data, batch_sampler=batch_sampler_train),
            'val': DataLoader(data, batch_sampler=batch_sampler_val)}

        torch.cuda.empty_cache()
        
        model = get_model(model_params)
        
        # # Load pretrained model
        # if hasattr(model_params, 'pretrained'):
        #         path_model = os.path.join(utils.utils_paths.mainpath.respath, model_params['pretrained'])
        #         saved_model_path = os.path.join(path_model, 'model_{}.pt'.format(k_ind))
        #         model.load_state_dict(torchload(saved_model_path), strict=False)
        
        if 'pretrained' in model_params:
            pretrained_models = model_params['pretrained']
            if pretrained_models[0] == 'AutoencoderCT_LR1e-4':
                path_model = os.path.join(utils.utils_paths.respath, pretrained_models[0])
                saved_model_path = os.path.join(path_model, 'model_{}.pt'.format(k_ind))
                state_dict = torchload(saved_model_path)
                new_state_dict = {}
                for name, param in state_dict.items():
                    if 'module.down_blocks' in name:
                        new_name = name.replace('module.down_blocks', 'module.down_blocks_CT')
                        new_state_dict[new_name] = param
                state_dict = new_state_dict
                model.load_state_dict(state_dict, strict=False)
            if pretrained_models[1] == 'AutoencoderCTCA_LR1e-4':
                path_model = os.path.join(utils.utils_paths.respath, pretrained_models[1])
                saved_model_path = os.path.join(path_model, 'model_{}.pt'.format(k_ind))
                state_dict = torchload(saved_model_path)
                new_state_dict = {}
                for name, param in state_dict.items():
                    if 'module.down_blocks' in name:
                        new_name = name.replace('module.down_blocks', 'module.down_blocks_CTCA')
                        new_state_dict[new_name] = param
                    elif 'module.up_blocks' in name:
                        new_name = name.replace('module.up_blocks', 'module.up_blocks_CTCA')
                        new_state_dict[new_name] = param
                state_dict = new_state_dict
                model.load_state_dict(state_dict, strict=False)
                
        
        # Define losses and acc
        lossfunc = getattr(utils.utils_trainAE, model_params['loss'])()
        #accfunc = utils.utils_trainAE.ClassAcc()
    
        valpred_fpath = os.path.join(path_results, 'val_examples')
        if not os.path.isdir(valpred_fpath):
                os.mkdir(valpred_fpath)
        valPredictions = utils.utils_trainAE.dataPredictions(fpath=valpred_fpath)
        if hasattr(model_params, 'n_imgs_per_epochs'):
                n_batches = int(model_params['n_imgs_per_epochs'] / model_params['batch_size'])
        else:
            n_batches = int(len(train_index) / model_params['batch_size'])
       
        torch.cuda.empty_cache()
        
        model, history = utils.utils_trainAE.train(model, dataloaders['train'], dataloaders['val'],
                                                   save_file_name, lossfunc,
                                               valPredictions=valPredictions,
                                               max_epochs_stop=model_params['max_epochs_stop'],
                                               n_epochs=model_params['n_epochs'],
                                               n_batches=n_batches,
                                               learning_rate=model_params['learning_rate'])
    
        with open(os.path.join(path_results, 'fold{}history.csv'.format(k_ind)), 'w',
                      newline='') as csvfile:
                filewriter = csv.writer(csvfile)
                filewriter.writerow(['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
                filewriter.writerows(history)

if __name__ == '__main__':
    augm_params = {'img_range': [-1000, 1000],
                   'crop': .11,
                   'rotate': 10,  
                   'flip': 0}
    model_params = {'model_folder': 'SparseAutoencoder_2E2D_LR1e-4_batch1',
                    'model_name': 'autoencoder_2E2D',
                    'batch_size': 1,
                    'size_input': 512,
                    'in_channels': 1,
                    'loss': 'SparseLoss',
                    'max_epochs_stop': 8,
                    'n_epochs': 100,
                    'learning_rate': 1e-4,
                    'train_paths': [os.path.join(utils.utils_paths.datapaths['cardiac_ct'], 'cardiac_ct_ctca1.csv')],
                    'nfolds': 5,
            
                    'augm_params': augm_params}
    
    model_params['pretrained'] = ['AutoencoderCT_LR1e-4', 'AutoencoderCTCA_LR1e-4']

    # Results paths
    path_results = os.path.join(utils.utils_paths.mainpath, 'results')
    main(model_params, path_results)