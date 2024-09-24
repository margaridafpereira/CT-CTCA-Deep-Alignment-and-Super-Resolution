# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:24:17 2024

@author: MargaridaP
"""
import numpy as np
import torch
import os
from pytorch_fid import fid_score

def calcular_fid_lote(imagens_reais, imagens_geradas, batch_size, device, dims):
    fid_values = []
    num_batches = len(imagens_reais) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        imagens_reais_lote = imagens_reais[start_idx:end_idx]
        imagens_geradas_lote = imagens_geradas[start_idx:end_idx]
        
        m1, s1 = fid_score.calculate_activation_statistics(imagens_reais_lote, model, batch_size, dims, device)
        m2, s2 = fid_score.calculate_activation_statistics(imagens_geradas_lote, model, batch_size, dims, device)
        
        fid_value = fid_score.calculate_frechet_distance(m1, s1, m2, s2)
        print(fid_value)
        fid_values.append(fid_value)
    return fid_values

caminho_imagens_reais = 'F:\\MargaridaP\\real_slices_png'
caminho_imagens_geradas = 'F:\\MargaridaP\\unet_slices_png'
arquivos_imagens_reais = [os.path.join(caminho_imagens_reais, f) for f in os.listdir(caminho_imagens_reais) if f.endswith('.npy')]
arquivos_imagens_geradas = [os.path.join(caminho_imagens_geradas, f) for f in os.listdir(caminho_imagens_geradas) if f.endswith('.npy')]

batch_size = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dims = 2048

block_idx = fid_score.InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = fid_score.InceptionV3([block_idx]).to(device)

fid_values = []
for i in range(0, len(arquivos_imagens_reais), batch_size):
    imagens_reais_batch = np.concatenate([np.load(f) for f in arquivos_imagens_reais[i:i+batch_size]], axis=0)
    imagens_geradas_batch = np.concatenate([np.load(f) for f in arquivos_imagens_geradas[i:i+batch_size]], axis=0)
    fid_values.extend(calcular_fid_lote(imagens_reais_batch, imagens_geradas_batch, batch_size, device, dims))

fid_mean = np.mean(fid_values)
print('FID Médio:', fid_mean)

