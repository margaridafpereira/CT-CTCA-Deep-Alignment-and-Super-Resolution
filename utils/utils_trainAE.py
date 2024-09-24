from torchvision import transforms, datasets, models
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim, cuda

import os, shutil, csv
# Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import copy
from PIL import Image

# Timing utility
from timeit import default_timer as timer

#import pytorch_ssim

class dataPredictions(object):
    def __init__(self, fpath='.'):
        self.predictions = []
        self.fpath = fpath
            
    def append(self, input, output, fnames=None):
        for ind, (i, o) in enumerate(zip(input, output)):
            i = np.squeeze(i.cpu().detach().numpy())
            o = np.squeeze(o.cpu().detach().numpy())
            # if target is None:
            #     t = np.zeros(i.shape)
            # else:
            #     t = np.squeeze(target[ind].cpu().detach().numpy())
            if fnames is None:
                rpath = f'{len(self.predictions)}'
            else:
                rpath = f'{os.path.splitext(os.path.split(fnames[ind])[-1])[0]}'

            self.predictions.append([i, o, rpath])
   

    def write(self, fpath=None, mode='output', nexamples=10, clear=False,  prediction_type=None):
        if fpath is None:
            if self.fpath is None:
                print('Could not write predictions to empty fpath.')
                return
            else:
                fpath = self.fpath

        
        for ind, (i, o, fname) in enumerate(self.predictions):
            i = ((i - np.min(i)) / (np.max(i) - np.min(i))) * 255
            i = np.clip(i, 0, 255).astype(np.uint8)
            
            o = ((o - np.min(o)) / (np.max(o) - np.min(o))) * 255
            o = np.clip(o, 0, 255).astype(np.uint8)
            
            # t = ((t - np.min(t)) / (np.max(t) - np.min(t))) * 255
            # t = np.clip(t, 0, 255).astype(np.uint8)
    
            if len(i.shape) > 2:
                i = i[i.shape[0]//2, :]
                o = o[o.shape[0]//2, :]
                # t = t[t.shape[0]//2, :, :]
    
            if mode == 'output':
                    
                np.save(os.path.join(fpath, f'{fname}_{prediction_type}.npy'), o)
                np.save(os.path.join(fpath, f'{fname}_input{prediction_type}.npy'), i)
            
            
            elif mode == 'examples':
                if len(i.shape) != len(o.shape):
                    print("Skipping case with incompatible dimensions")
                    continue
                
                if prediction_type is not None:
                    filename = f'{fname}_{prediction_type}.png'
                else:
                    filename = f'{fname}.png'

                Image.fromarray(np.concatenate((i, o), axis=1)).save(os.path.join(fpath, filename))
                if ind == nexamples - 1:
                    return

        if clear:
            self.predictions = []
      
class DataloaderIterator(object):
    def __init__(self, mode, calcLoss, dataloader, n_batches=None, datapredictions=dataPredictions()):
        self.mode = mode
        self.calcLoss = calcLoss
        #self.calcAcc = calcAcc
        self.dataloader = dataloader
        if n_batches is None:
            self.nbatches = len(dataloader)
        else:
            self.nbatches = min(n_batches, len(dataloader))
        
        self.predictionsCT = dataPredictions(fpath=datapredictions.fpath)
        self.predictionsCTCA = dataPredictions(fpath=datapredictions.fpath)
        self.predictionscONTRAST = dataPredictions(fpath=datapredictions.fpath)

    def __call__(self, model, optimizer, epoch):
        self.losstotal = 0
        self.lossCTCA = 0
        self.lossCT = 0
        #self.acc = 0
        self.nsamples = 0
        self.predictionsCT.predictions = []
        self.predictionsCTCA.predictions = []
        self.predictionscONTRAST.predictions = []

        
        self.start = timer()
        self.elapsed = 0
        
        for ii, (ct, ctca) in enumerate(self.dataloader):
            path_ct, path_ctca = self.dataloader.dataset.samples[ii]

            # if ii>10:
            #     break

            # # Useful for debbuging, can see the images and the weights and see whats going on, descomentar para ver as imagens a serem salvas
            # for d, t in zip(data, target):
            #     plt.figure()
            #     plt.imshow(d.numpy()[0, :, :])
            #     plt.figure()
            #     plt.imshow(t.numpy()[0, :, :])
            #     plt.show()

            # Tensors to gpu
            try:
                ct = ct.cuda()
                ctca = ctca.cuda()
                
                # batch_ct = batch_ct.cuda()
                # batch_ctca = batch_ctca.cuda()
                
                # ct = ct.to('cuda:1')
                # ctca = ctca.to('cuda:1')
                
                # batch_ct = batch_ct.to('cuda:1')
                # batch_ctca = batch_ctca.to('cuda:1')
            except:
                pass

            if self.mode == 'train':
                # Clear gradients
                optimizer.zero_grad()

            # Predicted output
            
            #output = model(ct, ctca)
            output = model(ct, ctca)
            
            #ctca_mask = (ctca == -1000).float()
           
            # Get loss:
            #lossCT = self.calcLoss(output, ct)
            torch.cuda.empty_cache()
            
            ct_enc = output[3]
            ctca_enc = output[4]
            
            out = output[0]+ output[1]
            
            losstotal= self.calcLoss(out, output[2], ct_enc, ctca_enc)
            lossCTCA = self.calcLoss(output[2], ctca,  ct_enc, ctca_enc)
            
            #output_ct = sampling(output[0], path_ct, path_ctca)# para média pesada
            output_ct = sampling1(output[0]) #para média direta
           
            output_ct = output_ct.cuda() 
            #output_ct = output_ct.to('cuda:1')
            
            lossCT= self.calcLoss(output_ct, ct,  ct_enc, ctca_enc)
            
            losstotal = losstotal * (1 - ((ctca == -1000).float()))
            lossCTCA = lossCTCA * (1 - ((ctca == -1000).float()))
            
            losstotal = losstotal.mean()
            lossCTCA = lossCTCA.mean()
            lossCT = lossCT.mean()
            
            torch.cuda.empty_cache()
            
            # Update the parameters
            if self.mode == 'train':
                losstotal.backward(retain_graph=True)                
                lossCTCA.backward(retain_graph=True)
                lossCT.backward()
                optimizer.step()
            
            torch.cuda.empty_cache()
            
            # Track train loss by multiplying average loss by number of examples in batch
            try:
                datasize = ct.size(0)
            except:
                datasize = ct[0].size(0)
            self.losstotal += losstotal.item() * datasize
            self.lossCTCA += lossCTCA.item() * datasize
            self.lossCT += lossCT.item() * datasize
            
            self.nsamples += datasize

            # Calculate accuracy
            #self.acc += self.calcAcc(output, ct)

            if self.mode == 'val':
                self.predictionsCT.append(input=ct, output = output[0])
                self.predictionsCTCA.append(input=ctca, output = output[2])
                self.predictionscONTRAST.append(input=ctca, output = output[1])

            # Track progress
            print(
                f'\rEpoch {self.mode}: {epoch}\t{100 * (ii + 1) / self.nbatches:.2f}% complete - loss contrast {self.losstotal / self.nsamples:.4f} - loss CTCA {self.lossCTCA / self.nsamples:.4f} - loss CT {self.lossCT / self.nsamples:.4f}. {timer() - self.start:.2f} seconds elapsed in epoch.',
                end='')
            if (ii + 1) == self.nbatches:
                break

        self.losstotal = self.losstotal / self.nsamples
        self.lossCTCA = self.lossCTCA / self.nsamples
        self.lossCT = self.lossCT / self.nsamples
        #self.acc = self.acc / self.nsamples
        print(f'\nEpoch: {epoch} \t{self.mode} Loss Contrast: {self.losstotal:.4f} \t{self.mode} Loss CTCA: {self.lossCTCA:.4f} \t{self.mode} Loss CT: {self.lossCT:.4f} \t{self.mode} ')
        self.elapsed = timer() - self.start

#MÉDIA DIRETA
def sampling1(ct):
    num_slices = ct.shape[2]

    if num_slices == 1:
        repeated_ct = ct.repeat(1, 1, 2, 1, 1)
        return repeated_ct
    elif num_slices == 2:
        return ct
    else:
        half_slices = num_slices // 2
        first_slice = torch.mean(ct[:, :, :half_slices, :, :], dim=2, keepdim=True)
        second_slice = torch.mean(ct[:, :, -(half_slices):, :, :], dim=2, keepdim=True)
        downsampled_ct = torch.cat([first_slice, second_slice], dim=2)
        return downsampled_ct


# #MÉDIA PESADA
def sampling(output_ct, path_ct, path_ctca):
    
    #print(output_ct.shape)
    num_slices = output_ct.shape[2]
    #print(num_slices)
   
    if num_slices == 2:
        return output_ct
        
    elif num_slices > 2:
        ct_filename = os.path.basename(path_ct)[:-4]
        ct_pos_path = os.path.join('D:\MargaridaP\pericardial_segmentation\data\CT_DICOM_extr\CT_pos', f'{ct_filename}.txt')
        ct_pos = np.loadtxt(ct_pos_path)
     
        ctca_filename = os.path.basename(path_ctca)[:-4]
        ctca_pos_path = os.path.join('D:\MargaridaP\pericardial_segmentation\data\CT_DICOM_extr\CTCA_aligned_pos', f'{ctca_filename}.txt')
        ctca_pos = np.loadtxt(ctca_pos_path)

        sampled_slices_first = []  
        sampled_slices_second = []  
        
        for i in range(len(ct_pos)):  
            differences = np.abs(ct_pos[i] - ctca_pos)      

            weights = 1 / (differences + 1e-6)  
            weights /= np.sum(weights)
            weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(output_ct.device)
            weighted_slices = output_ct * weights_tensor
                
            sampled_slice = torch.sum(weighted_slices, dim=0, keepdim=True)
                
            if i == 0:  
                    sampled_slices_first.append(sampled_slice[:, :, :1])
            elif i == 1:  
                    sampled_slices_second.append(sampled_slice[:, :, :1])
            
        sampled_ct_first = torch.cat(sampled_slices_first, dim=2) 
        sampled_ct_second = torch.cat(sampled_slices_second, dim=2)
   
        sampled_ct = torch.cat([sampled_ct_first, sampled_ct_second], dim=2)
        #print(sampled_ct.shape)
   
        return sampled_ct
    
# class ClassAcc(object):
#     def __call__(self, output, target):
#         pred = torch.round(output)
#         try:
#             target = target.cuda()
#         except:
#             pass
#         correct_tensor = pred.eq(target.data.view_as(pred))
#         return torch.mean(correct_tensor.type(torch.FloatTensor)).item() * output.size(0)

class ClassAcc(object):
    def __call__(self, pred, target):
        pred = pred.cpu().detach().numpy() if hasattr(pred, 'cpu') else pred
        target = target.cpu().detach().numpy() if hasattr(target, 'cpu') else target

        correct = np.sum(pred == target)  
        total = np.prod(pred.shape)

        accuracy = correct / total 

        return accuracy
  
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, outputs, targets):
        return F.mse_loss(outputs, targets) 

class SparseLoss(nn.Module):
    def __init__(self, factor=1.0, alpha=0.001):
        super(SparseLoss, self).__init__()
        self.factor = factor
        self.mse_loss = MSELoss()
        self.l1_loss = nn.L1Loss()
        self.alpha = alpha

    def forward(self, outputs, targets, ct_enc, ctca_enc):
        mse = self.mse_loss(outputs, targets)
        
        l1_penalty = self.alpha * (self.l1_loss(ct_enc, torch.zeros_like(ct_enc)) + self.l1_loss(ctca_enc, torch.zeros_like(ctca_enc)))
        loss = mse + l1_penalty

        return loss
    
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.ssim_module = pytorch_ssim.SSIM()

    def forward(self, outputs, targets):
        return 1 - self.ssim_module(outputs, targets)
    
class CombinedLoss(nn.Module):
    def __init__(self, factor=1.0):
        super(CombinedLoss, self).__init__()
        self.factor = 0.06/1100 #valor loss estabiliza SSIM e MSE
        self.mse_loss = MSELoss()
        self.ssim_loss = SSIMLoss()

    def forward(self, outputs, targets):
        mse = self.mse_loss(outputs, targets)
        ssim = self.ssim_loss(outputs, targets)
        combined_loss = self.factor*mse + ssim 
        return combined_loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return F.binary_cross_entropy(inputs, targets, reduction='mean')

    
def train(model, train_loader, valid_loader,
          save_file_name, lossCriterion,
          valPredictions=dataPredictions(), learning_rate=1e-4, 
          max_epochs_stop=3, n_epochs=20, n_batches=None):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Early stopping initialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    num_unfreeze_layer = 1
    history = []
    overall_start = timer()
    if not hasattr(model, 'epochs'):
        model.epochs = 0

    # Main loop
    trainIt = DataloaderIterator('train', lossCriterion, train_loader, n_batches=n_batches)
    valIt = DataloaderIterator('val', lossCriterion, valid_loader, n_batches=n_batches,
                               datapredictions=valPredictions)
    
    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        
        # Training loop
        model.train()  # Set to training mode
        trainIt(model, optimizer, epoch)
        model.epochs += 1

        # Validation loop
        with torch.no_grad():  # Don't need to keep track of gradients
            model.eval()  # Set to evaluation mode
            valIt(model, optimizer, epoch)

            # write history at the end of each epoch!
            history.append([trainIt.losstotal, valIt.losstotal, trainIt.lossCTCA, valIt.lossCTCA, trainIt.lossCT, valIt.lossCT])
            
            
            # Save the model if validation loss decreases
            if valIt.lossCT < valid_loss_min and  valIt.lossCTCA < valid_loss_min and valIt.losstotal < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_file_name)
                # Track improvement
                epochs_no_improve = 0
                valid_loss_minCT = valIt.lossCT
                valid_loss_minCTCA = valIt.lossCTCA
                valid_loss_mintotal = valIt.losstotal
                #valid_best_acc = valIt.acc
                best_epoch = epoch
                # Write predictions for best model
                #valIt.predictions.write(mode='examples')
                valIt.predictionsCT.write(mode='examples', prediction_type='CT')
                valIt.predictionsCTCA.write(mode='examples', prediction_type='CTCA')
                valIt.predictionscONTRAST.write(mode='examples', prediction_type='Contrast')

            else:  # Otherwise increment count of epochs with no improvement
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss CT: {valid_loss_minCT:.2f}, loss CTCA: {valid_loss_minCTCA:.2f}, loss Contrast: {valid_loss_mintotal:.2f}')
                    total_time = timer() - overall_start
                    print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.')
                    break

        # Report back minimum estimated time
        epochs_missing = max_epochs_stop - epochs_no_improve
        estimated_time = epochs_missing * (trainIt.elapsed + valIt.elapsed)
        print(f'Check back in {estimated_time:.2f} seconds.')

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(f'\nBest epoch: {best_epoch} with loss CT: {valid_loss_minCT:.2f}, loss CTCA: {valid_loss_minCTCA:.2f}, loss Contrast: {valid_loss_mintotal:.2f}')
    print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.')

    return model, history


def save_checkpoint(model, multi_gpu, path):
    """Save a PyTorch model checkpoint

    Params
    --------
        model (PyTorch model): model to save
        path (str): location to save model. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    model_name = path.split('-')[0]

    # Basic details
    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'epochs': model.epochs,
    }

    # Extract the final classifier and the state dictionary
    if 'vgg' in model_name or 'alexnet' in model_name or 'densenet' in model_name:
        # Check to see if model was parallelized
        if multi_gpu:
            checkpoint['classifier'] = model.module.classifier
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['classifier'] = model.classifier
            checkpoint['state_dict'] = model.state_dict()

    else:
        if multi_gpu:
            checkpoint['fc'] = model.module.fc
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['fc'] = model.fc
            checkpoint['state_dict'] = model.state_dict()

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)
