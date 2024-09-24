# -*- coding: utf-8 -*-
"""
Created on Tue May 21 23:04:23 2024

@author: MargaridaP
"""

# import os
# import numpy as np
# import pandas as pd
# from skimage.metrics import mean_squared_error

# def dice_coefficient(mask1, mask2):
#     intersection = np.sum(mask1 & mask2)
#     union = np.sum(mask1) + np.sum(mask2)
#     dice = (2.0 * intersection) / union if union > 0 else 0.0
#     return dice

# # Diretórios de entrada
# #ct_folder_base = 'D:\\MargaridaP\\pericardial_segmentation\\data\\results\\UNet_reconstruct_MSELoss_LR1e-4\\MargaridaP_test'
# ct_folder_base = 'F:\\MargaridaP\\interpolation_SR\\interpolated_slice'
# ctca_aligned_folder = 'F:\\MargaridaP\\CTCA_aligned'

# # Arquivo de resultados
# excel_file_path = 'SR_Interpolation_results_dice.xlsx'

# if os.path.exists(excel_file_path):
#     dice_results_df = pd.read_excel(excel_file_path)
# else:
#     dice_results_df = pd.DataFrame()
    
# for filename in os.listdir(ct_folder_base):
#         if filename.endswith('_reconstructed.npy'):
#             ct_path = os.path.join(ct_folder_base, filename)
#             ctca_path = os.path.join(ctca_aligned_folder, filename)
            
#             if os.path.exists(ctca_path):
#                 ct_data = np.load(ct_path).astype(np.float32)  
#                 ctca_data = np.load(ctca_path).astype(np.float32)  
                
#                 threshold = 0
#                 # ct_data[ct_data >= threshold] = np.nan
#                 # ctca_data[ctca_data >= threshold] = np.nan

#                 # mask = ~np.isnan(ct_data) & ~np.isnan(ctca_data)
#                 # mse = mean_squared_error(ct_data[mask], ctca_data[mask])
                
#                 # new_result = pd.DataFrame({
#                 #     'Fold': [fold],
#                 #     'CT': [filename],
#                 #     'CTCA': [filename],
#                 #     'MSE': [mse]
#                 # })

#                 # mse_results_df = pd.concat([mse_results_df, new_result], ignore_index=True)
                
#                 # Calcular DICE (se necessário)
#                 ct_mask = ct_data < threshold
#                 ctca_mask = ctca_data < threshold
#                 dice = dice_coefficient(ct_mask, ctca_mask)
                
#                 new_result = pd.DataFrame({
#                     'CT': [filename],
#                     'CTCA': [filename],
#                     'DICE': [dice]
#                 })
                
#                 dice_results_df = pd.concat([dice_results_df, new_result], ignore_index=True)
#             else:
#                 print(f"Arquivo CTCA correspondente não encontrado para {filename}")

# #mse_results_df.to_excel(excel_file_path, index=False)
# dice_results_df.to_excel(excel_file_path, index=False)

# print("Cálculo do DICE concluído.")

import os
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error

def dice_coefficient(mask1, mask2):
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1) + np.sum(mask2)
    dice = (2.0 * intersection) / union if union > 0 else 0.0
    return dice

# Diretórios de entrada
#ct_folder_base = 'D:\\MargaridaP\\pericardial_segmentation\\data\\results\\FinalUNet_MSELoss_lr10-4\\MargaridaP_test'#unet SR
ct_folder_base = 'F:\\MargaridaP\\interpolation_SR\\interpolated_slice' #INT SR

ctca_aligned_folder = 'F:\\MargaridaP\\CTCA_aligned'

# Arquivo de resultados
excel_file_path = 'F:\MargaridaP\DICE\DICE_unetSR.xlsx'

if os.path.exists(excel_file_path):
    dice_results_df = pd.read_excel(excel_file_path)
else:
    dice_results_df = pd.DataFrame()
    
for filename in os.listdir(ct_folder_base):
    if filename.endswith('_reconstructed.npy'):
        ct_path = os.path.join(ct_folder_base, filename)
        ctca_path = os.path.join(ctca_aligned_folder, filename)
        
        if os.path.exists(ctca_path):
            try:
                ct_data = np.load(ct_path).astype(np.float32)  
                ctca_data = np.load(ctca_path).astype(np.float32)  

                threshold = 0
                # ct_data[ct_data <= threshold] = np.nan
                # ctca_data[ctca_data <= threshold] = np.nan
                ct_mask = ct_data < threshold
                ctca_mask = ctca_data < threshold
                dice = dice_coefficient(ct_mask, ctca_mask)
                #print (dice)
                
                new_result = pd.DataFrame({
                    'CT': [filename],
                    'CTCA': [filename],
                    'DICE': [dice]
                })
                
                dice_results_df = pd.concat([dice_results_df, new_result], ignore_index=True)
            except ValueError as e:
                print(f"Ignorando arquivo {filename} devido a um erro: {e}")
        else:
            print(f"Arquivo CTCA correspondente não encontrado para {filename}")

dice_results_df.to_excel(excel_file_path, index=False)

print("Cálculo do DICE concluído.")

# # -*- coding: utf-8 -*-
# """
# Created on Tue May 21 23:04:23 2024

# @author: MargaridaP
# """

# import os
# import numpy as np
# import pandas as pd
# from skimage.metrics import mean_squared_error

# def dice_coefficient(mask1, mask2):
#     intersection = np.sum(mask1 & mask2)
#     union = np.sum(mask1) + np.sum(mask2)
#     dice = (2.0 * intersection) / union if union > 0 else 0.0
#     return dice

# # Diretórios de entrada
# ct_folder_base = 'F:\\MargaridaP\\interpolation_SR\\interpolated_slice'
# ctca_aligned_folder = 'F:\\MargaridaP\\CTCA_aligned'

# # Arquivo de resultados
# excel_file_path = 'InterpolationSR_results_dice.xlsx'

# if os.path.exists(excel_file_path):
#     dice_results_df = pd.read_excel(excel_file_path)
# else:
#     dice_results_df = pd.DataFrame()

# for filename in os.listdir(ct_folder_base):
#     if filename.endswith('_reconstructed.npy'):
#         ct_path = os.path.join(ct_folder_base, filename)
#         ctca_path = os.path.join(ctca_aligned_folder, filename)
        
#         if os.path.exists(ctca_path):
#             try:
#                 ct_data = np.load(ct_path).astype(np.float32)  
#                 ctca_data = np.load(ctca_path).astype(np.float32)  
#             except ValueError as e:
#                 if "cannot reshape array of size 0 into shape" in str(e):
#                     print(f"Erro ao carregar {filename}: {e}")
#                     continue
            
#             # threshold = 0
#             # ct_data[ct_data >= threshold] = np.nan
#             # ctca_data[ctca_data >= threshold] = np.nan
              
#             threshold = 0
#             ct_mask = ct_data < threshold
#             ctca_mask = ct_data < threshold
            
#             dice = dice_coefficient(ct_mask, ctca_mask)
            
#             new_result = pd.DataFrame({
#                 'CT': [filename],
#                 'CTCA': [filename],
#                 'DICE': [dice]
#             })
            
#             dice_results_df = pd.concat([dice_results_df, new_result], ignore_index=True)
#         else:
#             print(f"Arquivo CTCA correspondente não encontrado para {filename}")

# dice_results_df.to_excel(excel_file_path, index=False)

# print("Cálculo do DICE concluído.")
