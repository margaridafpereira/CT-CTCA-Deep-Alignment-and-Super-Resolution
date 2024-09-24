# -- coding: utf-8 --
"""
Created on Sun Nov  5 22:00:53 2023
@author: Margarida
"""
import os
import nrrd
from align_data_on_spatial_pos import interpolate_CTCAnew
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
import numpy as np
from align_data_on_spatial_pos import matching_CTandCTCA_slices
 
input_folder = 'F:\MargaridaP\CT_DICOM_extr_original'

for folder in os.listdir(input_folder):
    total_mse = []
    total_dice = 0        
    num_slices = 0
    squared_diff_sum = 0
    
    folder_path = os.path.join(input_folder, folder)

    if os.path.isdir(folder_path):
        
        #Para carregar as CT interpoladas
        CT_folder = os.path.join(folder_path, 'interpolationSR')
        for filename in os.listdir(CT_folder):
            if filename.endswith('.nrrd'):
                ct_path = os.path.join(CT_folder, filename)
                print(ct_path)
            # elif filename.endswith('info'):
            #     ct_info= os.path.join(CT_folder, filename)
                
        #Para carregar as CT originais
        #ncct = ''
        ctca = ''
        for f in os.listdir(folder_path):
            # if f.split('_')[0] == 'CT':
            #     ncct = f
            #     ct_path = os.path.join(folder_path, ncct)
            if f.split('_')[0] == 'CTCA':
                ctca = f
                ctca_path = os.path.join(folder_path, ctca)
                print(ctca_path)

        #USAR SÓ QUANDO QUISER GUARDAR CTCA ALINHADAS
        # output_path = os.path.join(folder_path, 'newCTCA')
        
        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)       
        
        data_ct, header_ct = nrrd.read(ct_path)

        matching_slices_indices = matching_CTandCTCA_slices(ct_path, ctca_path)
        
        for z_ind_ct in matching_slices_indices:    
            #ct_slice, z_ind_ct, ctca_new, ctca_new_filename = interpolate_CTCAnew(ct_path, z_ind_ct, ctca_path, output_path)
            ct_slice, ctca_new = interpolate_CTCAnew(ct_path, z_ind_ct, ctca_path)
                   
            #plt.imsave(ctca_new_filename, ctca_new, format='png')
            
              #MSE --> threshold <0 HU--------------------------------------------------------------------------------------
            threshold = 0
            ct_slice[ct_slice >= threshold] = np.nan #None
            ctca_new[ctca_new >= threshold] = np.nan #None
    
            # plt.figure(figsize=(8, 8))
            # plt.subplot(1, 2, 1)
            # plt.imshow(ct_slice, cmap='gray')
            # plt.title("CT")
            # plt.axis('off')
            
            # plt.subplot(1, 2, 2)
            # plt.imshow(ctca_new, cmap='gray')
            # plt.title("CTCA")
            # plt.axis('off')
            
            # Save threshold CT image
            # ct_folder = os.path.join(folder_path, 'threshold', 'CT')
            # if not os.path.exists(ct_folder):
            #     os.makedirs(ct_folder)
            # ct_slice_filename = os.path.join(ct_folder, f'{folder}ct_slice{z_ind_ct}_thresholded.png')
            # plt.imsave(ct_slice_filename, ct_slice, cmap='gray', format='png')
            
            # # Save thresholded CTCA image
            # ctca_folder = os.path.join(folder_path, 'threshold', 'CTCA')
            # if not os.path.exists(ctca_folder):
            #     os.makedirs(ctca_folder)
            # ctca_new_filename = os.path.join(ctca_folder, f'{folder}ctca_new{z_ind_ct}_thresholded.png')
            # plt.imsave(ctca_new_filename, ctca_new, cmap='gray', format='png')
            
            # mse = mean_squared_error(ct_slice, ctca_new)
            # print(f"MSE between ct_slice and ctca_new: {mse}")
            
            mask = ~np.isnan(ct_slice) & ~np.isnan(ctca_new)
            mse = mean_squared_error(ct_slice[mask], ctca_new[mask])
            #print(f"MSE between ct_slice and ctca_new: {mse}")
            
            total_mse.append(mse) 
            
            # Save MSE values to a text file
            mse_file_path = os.path.join(folder_path, 'threshold', 'mse_values.txt')
            if not os.path.exists(os.path.dirname(mse_file_path)):
                os.makedirs(os.path.dirname(mse_file_path))
                
            with open(mse_file_path, 'a') as mse_file:
                mse_file.write(f"Slice {z_ind_ct}: {mse}\n")
        
        avg_mse = np.mean(total_mse)
        mse_std_dev = np.std(total_mse)
        
        with open(mse_file_path, 'r') as old_file:
            old_data = old_file.read()
            
        with open(mse_file_path, 'w') as mse_file:
            mse_file.write(f"Mean MSE: {avg_mse}\n\n")
            mse_file.write(f"Standart Deviation: {mse_std_dev}\n\n")
            mse_file.write(old_data)
#---------------------------------------------------------------------------------------------------------------------------------------------------
    
        #     #DICE 
        #     def dice_coefficient(mask1, mask2):
        #         intersection = np.sum(mask1 & mask2)
        #         union = np.sum(mask1) + np.sum(mask2)
        #         dice = (2.0 * intersection) / union if union > 0 else 0.0
        #         return dice
            
        #     threshold = 0
        #     ct_slice_mask = ct_slice >= threshold
        #     ctca_new_mask = ctca_new >= threshold
            
        #     dice_coefficient = dice_coefficient(ct_slice_mask, ctca_new_mask)
        #     #print(f"Dice coefficient between CT and CTCA regions: {dice_coefficient}")
            
        #     total_dice += dice_coefficient
            
        #     dice_file_path = os.path.join(folder_path, 'threshold1', 'dice_coefficients.txt')
        #     with open(dice_file_path, 'a') as dice_file:
        #         dice_file.write(f"Slice {z_ind_ct}: {dice_coefficient}\n")

        # avg_dice = total_dice / num_slices
        
        # with open(dice_file_path, 'r') as old_dice_file:
        #     old_dice = old_dice_file.read()
        # with open(dice_file_path, 'w') as dice_file:
        #     dice_file.write(f"Mean DICE: {avg_dice}\n\n")
        #     dice_file.write(old_dice)