# PyTorch
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
from matplotlib import pyplot as plt

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import sys

sys.path.insert(0, '..')
import utils.utils_dataAE_CT_CTCA
from utils.utils_model import get_model, save_model_params, load_model_params
from utils.utils_transforms2 import CTTransform
from utils.utils_csv import readCsvDataLists
from utils.utils_trainAE_CT_CTCA import dataPredictions
import utils.utils_paths


def main(model_folder, datasetlist, subset=None, dispFlag=False):
    # Load model params
    model_params = load_model_params(model_folder)

    nmodels = 0
    nfolds = model_params['nfolds']
    for k_ind in range(nfolds):
        if os.path.isfile(os.path.join(model_folder, 'model_.pt'.format(k_ind))):
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
                # Replacing testfold with subset in the getDataLoader function call
                dataloader = getDataLoader(dataset, datastr, subset, k_ind, nfolds)
                #dataloader = getDataLoader(dataset, datastr, testfold, k_ind, nfolds)
                predict(model, dataloader, path_results, k_ind, dispFlag=dispFlag)
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
                predict(model, dataloader, path_results, k_ind, dispFlag=dispFlag)


# def loadData(datastr, image_transform, model_folder, subset):
#     datafolder = os.path.split(os.path.split(datastr)[0])[-1]
#     if subset is None:
#         path_results = os.path.join(model_folder, datafolder)
#     else:
#         path_results = os.path.join(model_folder, datafolder + '_' + subset)

#     if not os.path.isdir(path_results):
#         os.mkdir(path_results)

#     # Datasets from each folder
#     dataset = utils.utils_dataAE_CT_CTCA.DatasetCSV(root=datastr, transform=image_transform)
#     print('Found {} samples.'.format(len(dataset.samples)))

#     return dataset, path_results

def loadData(datastr, image_transform, model_folder, subset):
    datafolder = os.path.split(os.path.split(datastr)[0])[-1]
    if subset is None:
        path_results = os.path.join(model_folder, datafolder)
    else:
        path_results = os.path.join(model_folder, datafolder + '_' + subset)

    if not os.path.isdir(path_results):
        os.mkdir(path_results)

    # Datasets from each folder
    dataset = utils.utils_dataAE_CT_CTCA.DatasetCSV(root=datastr, transform=image_transform)
    print('Found {} samples.'.format(len(dataset.samples)))

    return dataset, path_results



def loadModel(model_folder, k_ind, model_params):
    model_path = os.path.join(model_folder, 'model_{}.pt'.format(k_ind))
    print('Loading', model_path)
    # Load model
    model = get_model(model_params)
    model_dict = torchload(model_path)
    model_dict = {k: model_dict[k] for k in model.state_dict()}
    model.load_state_dict(model_dict)

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


def predict(model, dataloader, path_results, k_ind, dispFlag=False):
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

            output = model(data, target)

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
            #dPredictions.write(path_results_k, clear=True, prediction_type=None)

    print(
        f'Fold {k_ind}: \t{100 * (ii + 1) / nbatches:.2f}% complete. {time.time() - st_time:.2f} seconds elapsed in fold.')


# if __name__ == '__main__':
#     model_folder = os.path.join(utils.utils_paths.mainpath, 'results', 'AutoencoderCT_CTCA_unet_lr10-4')
#     datasetlist = [os.path.join(utils.utils_paths.datapaths['cardiac_ct'], 'cardiac_ct_ctcaAE_unet.csv')],
#     main(model_folder, datasetlist, subset='test', dispFlag=False)
    
if __name__ == '__main__':
    model_folder = os.path.join(utils.utils_paths.mainpath, 'results', '2AutoencoderCT_CTCA_unet_lr10-4')
    datasetlist = [os.path.join(utils.utils_paths.datapaths['cardiac_ct'], 'cardiac_ct_ctcaAE_unet.csv')]
    main(model_folder, datasetlist, subset='test', dispFlag=False)

