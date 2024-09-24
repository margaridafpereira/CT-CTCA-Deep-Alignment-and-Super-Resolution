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

import pytorch_ssim


class dataPredictions(object):
    def __init__(self, fpath='.'):
        self.predictions = []
        self.fpath = fpath
            
    def append(self, input, output, target=None, fnames=None):
        for ind, (i, o) in enumerate(zip(input, output)):
            i = np.squeeze(i.cpu().detach().numpy())
            o = np.squeeze(o.cpu().detach().numpy())
            if target is None:
                t = np.zeros(i.shape)
            else:
                t = np.squeeze(target[ind].cpu().detach().numpy())
            if fnames is None:
                rpath = f'{len(self.predictions)}'
            else:
                rpath = f'{os.path.splitext(os.path.split(fnames[ind])[-1])[0]}'

            self.predictions.append([i, o, t, rpath])
                
    def write(self, fpath=None, mode='output', nexamples=10, clear=False):
        if fpath is None:
            if self.fpath is None:
                print('Could not write predictions to empty fpath.')
                return
            else:
                fpath = self.fpath

        for ind, (i, o, t, fname) in enumerate(self.predictions):
            i = ((i - np.min(i)) / (np.max(i) - np.min(i))) * 255
            i = i.astype(np.uint8)
            o = ((o - np.min(o)) / (np.max(o) - np.min(o))) * 255
            o = o.astype(np.uint8)
            t = ((t - np.min(t)) / (np.max(t) - np.min(t))) * 255
            t = t.astype(np.uint8)

            if len(i.shape) > 2:
                i = i[i.shape[0]//2, :, :]

            if mode == 'output':
                #np.save(os.path.join(fpath, f'{fname}.npy'), i)
                np.save(os.path.join(fpath, f'{fname}_reconstructed.npy'), o)
                np.save(os.path.join(fpath, f'{fname}_target.npy'), t)
            elif mode == 'examples':
                Image.fromarray(np.concatenate((i, o, t), axis=1)).save(os.path.join(fpath, f'{fname}.png'))
                #Image.fromarray(np.concatenate((i[0], i[1], o[0], o[1]), axis=1)).save(os.path.join(fpath, f'{fname}.png'))
                if ind == nexamples - 1:
                    return

        if clear:
            self.predictions = []
            

class DataloaderIterator(object):
    def __init__(self, mode, calcLoss, calcAcc, dataloader, n_batches=None, datapredictions=dataPredictions()):
        self.mode = mode
        self.calcLoss = calcLoss
        self.calcAcc = calcAcc
        self.dataloader = dataloader
        if n_batches is None:
            self.nbatches = len(dataloader)
        else:
            self.nbatches = min(n_batches, len(dataloader))
        self.predictions = datapredictions

    def __call__(self, model, optimizer, epoch):
        self.loss = 0
        self.acc = 0
        self.nsamples = 0
        self.predictions.predictions = []
        self.start = timer()
        self.elapsed = 0
        for ii, (data, target, pos_ct1, pos_ct2, pos_x) in enumerate(self.dataloader):
        
            #print(ii)

            # # Useful for debbuging, can see the images and the weights and see whats going on, descomentar para ver as imagens a serem salvas
            # for d, t in zip(data, target):
            #     plt.figure()
            #     plt.imshow(d.numpy()[0, :, :])
            #     plt.figure()
            #     plt.imshow(t.numpy()[0, :, :])
            #     plt.show()

            # Tensors to gpu
            try:
                data = data.cuda()
                target = target.cuda()
            except:
                pass

            if self.mode == 'train':
                # Clear gradients
                optimizer.zero_grad()

            # Predicted output
            output = model(data, pos_ct1, pos_ct2, pos_x)

            # Get loss:
            loss = self.calcLoss(output, target)

            # Update the parameters
            if self.mode == 'train':
                loss.backward()
                optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            try:
                datasize = data.size(0)
            except:
                datasize = data[0].size(0)
            self.loss += loss.item() * datasize
            self.nsamples += datasize

            # Calculate accuracy
            self.acc += self.calcAcc(output, target)

            if self.mode == 'val':
                self.predictions.append(data, output, target=target)

            # Track progress
            print(
                f'\rEpoch {self.mode}: {epoch}\t{100 * (ii + 1) / self.nbatches:.2f}% complete - loss {self.loss / self.nsamples:.4f}. {timer() - self.start:.2f} seconds elapsed in epoch.',
                end='')
            if (ii + 1) == self.nbatches:
                break

        self.loss = self.loss / self.nsamples
        self.acc = self.acc / self.nsamples
        print(f'\nEpoch: {epoch} \t{self.mode} Loss: {self.loss:.4f} \t{self.mode} Acc: {self.acc:.4f}')
        self.elapsed = timer() - self.start

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
          save_file_name, lossCriterion, accCriterion,
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
    trainIt = DataloaderIterator('train', lossCriterion, accCriterion, train_loader, n_batches=n_batches)
    valIt = DataloaderIterator('val', lossCriterion, accCriterion, valid_loader, n_batches=n_batches,
                               datapredictions=valPredictions)
    
    for epoch in range(n_epochs):
        # Training loop
        model.train()  # Set to training mode
        trainIt(model, optimizer, epoch)
        model.epochs += 1

        # Validation loop
        with torch.no_grad():  # Don't need to keep track of gradients
            model.eval()  # Set to evaluation mode
            valIt(model, optimizer, epoch)

            # write history at the end of each epoch!
            history.append([trainIt.loss, valIt.loss, trainIt.acc, valIt.acc])

            # Save the model if validation loss decreases
            if valIt.loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_file_name)
                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = valIt.loss
                valid_best_acc = valIt.acc
                best_epoch = epoch
                # Write predictions for best model
                valIt.predictions.write(mode='examples')
            else:  # Otherwise increment count of epochs with no improvement
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%')
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
    print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%')
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
