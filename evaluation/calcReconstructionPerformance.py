# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:50:33 2023

@author: MargaridaP
"""

from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio
import os, sys
import csv
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from skimage.measure import label
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import torch
import pytorch_ssim

sys.path.append('..')

#import CXR 
import utils.utils_csv
import utils.utils_paths 
from evalfuncs import plotPRC, plotROC
from utils.utils_data import npy_loader 

def getModelPredictions(fpath):
    tgt, pred = [],[]
    pred_name = []
    for fname in os.listdir(fpath):
        if fname.split('_')[-1] == 'reconstructed.npy':
            fname_target = fname.replace('_reconstructed.','_target.')
            if os.path.isfile(os.path.join(fpath,fname_target)):
                pred.append(np.array(npy_loader(os.path.join(fpath, fname))))
                tgt.append(np.array(npy_loader(os.path.join(fpath, fname_target))))
                pred_name.append(fname)
            else:
                print(f'No target found for {fname}.')

    print(f'{len(tgt)} CXR reconstructed found for {fpath}.')
    return tgt,pred,pred_name

def calcMSE(img1, img2):
    return mean_squared_error(img1, img2)

# def calcSSIM(img1, img2):
#     return structural_similarity(img1, img2, multichannel=True)

def calcSSIM(img1, img2):
    img1_tensor = torch.tensor(img1, dtype=torch.float32).unsqueeze(0) 
    img2_tensor = torch.tensor(img2, dtype=torch.float32).unsqueeze(0)

    ssim_module = pytorch_ssim.SSIM()
    ssim_value = ssim_module(img1_tensor, img2_tensor)

    return ssim_value.item()

def calcPSNR(img1, img2):
    max_pixel = 512.0
    return peak_signal_noise_ratio(img1, img2, data_range=max_pixel)

def createTrainValLossGraphs(modelPath):
    fold_names = ['fold1history', 'fold2history', 'fold3history', 'fold4history', 'fold5history']

    for fold_name in fold_names:
        name_csv = modelPath + '\\' + fold_name + '.csv'

        history_csv = pd.read_csv(name_csv)

        training_loss = history_csv['train_loss']
        validation_loss = history_csv['valid_loss']

        training_accuracy = history_csv['train_acc']
        validation_accuracy = history_csv['valid_acc']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        ax1.plot(training_loss, 'r--')
        ax1.plot(validation_loss, 'b-')
        ax1.legend(['Training Loss', 'Validation Loss'])

        ax2.plot(training_accuracy, 'r--')
        ax2.plot(validation_accuracy, 'b-')
        ax2.legend(['Training Sensitivity', 'Validation Sensitivity'])

        plt.title('LossAcc ' + fold_name)
        graphName = dset + 'lossAcc' + fold_name
        resultsGraphsFolder = os.path.join(model_folder[1], 'results_graphs')
        plt.savefig(os.path.join(resultsGraphsFolder, graphName))
        plt.show()

        return [training_loss, validation_loss, training_accuracy, validation_accuracy]

def extract_info_from_filename(filename):
    parts = filename.split('_')
    patient_id = '_'.join(parts[:3])
    image_name = parts[3]
    slice_num = int(parts[-2])
    return patient_id, image_name, slice_num

if __name__ == '__main__':
    model_folder = [os.path.join(utils.utils_paths.mainpath, 'results', 'UNet_SSIMLoss_LR1e-4')]

    dset = 'CT_DICOM_extr'
    subset = 'test'
    dset = dset+'_'+subset

    #k_folds = [0,1,2,3,4] 
    for k in k_folds:
    #k=2   
        results_folder = [os.path.join(mfolder, dset) for mfolder in model_folder]
        res_fpath = os.path.join(results_folder[0], 'fold{}predictions'.format(k))#k
        tgt, pred, pred_name = getModelPredictions(res_fpath)
        
        metrics_data = []
        
        for t, p, filename in zip(tgt, pred, pred_name):
                patient_id, image_name, slice_num = extract_info_from_filename(filename)
        
                mse = calcMSE(t, p)
                ssim = calcSSIM(t, p)
                psnr = calcPSNR(t, p)
        
                metrics_data.append({
                    'Patient_ID': patient_id,
                    'Image': image_name,
                    'Slice': slice_num,
                    'MSE': mse,
                    'SSIM': ssim,
                    'PSNR': psnr
                })
        
        df = pd.DataFrame(metrics_data)
        df.to_excel(os.path.join(model_folder[0], 'metrics_fold{}.xlsx'.format(k)), index=False)#k
        
        # Creates the graph with the train and validation loss and accuracy across the epochs for each fold
        #createTrainValLossGraphs(model_folder[0])
      
     # if __name__ == '__main__':
     #    model_folder = [os.path.join(utils.utils_paths.mainpath, 'results', 'UNet_reconstruct_SSIMLoss')]
     
     #    dset = 'CT_DICOM_extr'
     #    subset = 'test'
     #    dset = dset+'_'+subset
    
     #    #k_folds = [1, 2, 3, 4, 5]
        
            
     #    #for k in k_folds:
            
     #    results_folder = [os.path.join(mfolder, dset) for mfolder in model_folder]
     #    res_fpath = os.path.join(results_folder[0], 'fold{}predictions'.format(0))#k
     #    tgt, pred, pred_name = getModelPredictions(res_fpath)
    
     #    mse_scores = []
     #    ssim_scores = []
     #    psnr_scores = []
    
        # for t, p in zip(tgt, pred):
        #         mse = calcMSE(t, p)
        #         ssim = calcSSIM(t, p)
        #         psnr = calcPSNR(t, p)
    
        #         mse_scores.append(mse)
        #         ssim_scores.append(ssim)
        #         psnr_scores.append(psnr)
    
        # mean_mse = np.mean(mse_scores)
        # mean_ssim = np.mean(ssim_scores)
        # mean_psnr = np.mean(psnr_scores)
    
        # std_mse = np.std(mse_scores)
        # std_ssim = np.std(ssim_scores)
        # std_psnr = np.std(psnr_scores)
    
        # with open(os.path.join(model_folder[0], '{}_metrics.txt'.format(dset)), mode='a') as file:
        #         file.write(f"Fold {0} - Mean MSE: {mean_mse}, Mean SSIM: {mean_ssim}, Mean PSNR: {mean_psnr}\n")#k
        #         file.write(f"Fold {0} - Std MSE: {std_mse}, Std SSIM: {std_ssim}, Std PSNR: {std_psnr}\n")#k 
    
