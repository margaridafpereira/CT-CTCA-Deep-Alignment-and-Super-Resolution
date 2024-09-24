# # -*- coding: utf-8 -*-
# """
# Created on Wed Oct 25 23:24:24 2023

# @author: Margarida
# """

# import nrrd
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# input_folder = 'F:\MargaridaP\CT_DICOM_extr_original'

# def weighted_average_interpolation(slice1, slice2, distance1, distance2):
#     interpolated_slice = (distance2 * slice1 + distance1 * slice2) / (distance1 + distance2)
#     return interpolated_slice

# def get_last_CTCA_image(folder_path):
#     largest_x = -1
#     largest_x_filename = None

#     for filename in os.listdir(folder_path):
#         if filename.startswith('CTCA_') and filename.endswith('.nrrd'):
#             x = int(filename.split('_')[1].split('.')[0])
#             if x > largest_x:
#                 largest_x = x
#                 largest_x_filename = filename #nome da última CTCA-- ou seja a com maior resolução

#     return os.path.join(folder_path, largest_x_filename)

# for folder in os.listdir(input_folder):
#         folder_path = os.path.join(input_folder, folder)

#         if os.path.isdir(folder_path):
#             ct_filename = None
#             ctca_filename = get_last_CTCA_image(folder_path)

#             for filename in os.listdir(folder_path):
#                 if filename.startswith('CT_') and filename.endswith('.nrrd'):
#                     ct_filename = os.path.join(folder_path, filename)

#                 if ct_filename and ctca_filename:
#                     ct_data, ct_header = nrrd.read(ct_filename)
#                     ctca_data, ctca_header = nrrd.read(ctca_filename)
    
#                     num_slices_ct = ct_data.shape[2]
#                     space_directions_ct = ct_header['space directions']
#                     slice_thickness_ct = space_directions_ct[2, 2]
    
#                     num_slices_ctca = ctca_data.shape[2]
#                     space_directions_ctca = ctca_header['space directions']
#                     slice_thickness_ctca = space_directions_ctca[2, 2]
    
#                     z_positions_ct = [slice_thickness_ct * z for z in range(num_slices_ct)]
#                     z_positions_ctca = [slice_thickness_ctca * z for z in range(num_slices_ctca)]
    
#                     interval_start = min(z_positions_ct[:2])
#                     interval_end = max(z_positions_ct[:2])
    
#                     ctca_slices_in_interval = [z for z in z_positions_ctca if interval_start < z < interval_end]
    
#                     # if ctca_slices_in_interval:
#                     #     first_slice_z_ct = z_positions_ct[0]
#                     #     closest_z_ctca = min(ctca_slices_in_interval, key=lambda x: abs(x - first_slice_z_ct))
#                     #     distance = abs(closest_z_ctca - first_slice_z_ct)
    
#                     if ctca_slices_in_interval:
#                         first_slice_z_ct = z_positions_ct[0]
    
#                         distances_to_slice1 = [abs(z - first_slice_z_ct) for z in ctca_slices_in_interval]
    
#                     # INTERPOLAÇÃO
#                     distance_between_provided_slices = 2 * slice_thickness_ct
    
#                     interpolated_slices = []
#                     interpolated_z_positions = []
#                     interpolated_xy_positions = []
    
#                     for i in range(0, num_slices_ct - 2, 1):
#                         slice1 = ct_data[:, :, i]
#                         slice2 = ct_data[:, :, i + 2]
    
#                         print(len(ctca_slices_in_interval))
#                         for j in range(len(ctca_slices_in_interval)):
#                             distance_to_slice2 = distance_between_provided_slices - distances_to_slice1[j]
#                             interpolated_slice = weighted_average_interpolation(slice1, slice2, distances_to_slice1[j], distance_to_slice2)
#                             interpolated_slices.append(interpolated_slice)
    
#                             z_position_interpolated = z_positions_ct[0] + i * slice_thickness_ct + distances_to_slice1[j]
#                             x_position_interpolated = ct_header['space origin'][0]
#                             y_position_interpolated = ct_header['space origin'][1]
    
#                             interpolated_z_positions.append(z_position_interpolated)
#                             interpolated_xy_positions.append((x_position_interpolated, y_position_interpolated))
                            
#                             # output_folder_int = os.path.join(folder_path, 'interpolated_imageSR', f'interpolated_slices_{filename[:-5]}')
#                             # if not os.path.exists(output_folder_int):
#                             #      os.makedirs(output_folder_int)
        
#                             # output_filename_int = os.path.join(output_folder_int, f'interpolated__slice{i}_{filename[:-5]}.png')
#                             # plt.imsave(output_filename_int, interpolated_slice, cmap='gray')
                    
#                         output_folder = os.path.join(folder_path, 'interpolation_SR')
#                         if not os.path.exists(output_folder):
#                             os.makedirs(output_folder)
                            
#                         interpolated_volume = np.dstack(interpolated_slices)
#                         print(interpolated_volume.shape)
                        
#                         output_filename = os.path.join(output_folder, f'interpolated_{os.path.basename(ct_filename)}')
#                         nrrd.write(output_filename, interpolated_volume, header=ct_header)
        
#                         info_filename = os.path.join(output_folder, f'interpolated_{os.path.basename(ct_filename).replace(".nrrd", "_info.txt")}')
#                         with open(info_filename, 'w') as info_file:
#                             info_file.write("Slice Position (z),X Position, Y Position\n")
#                             for z, (x, y) in zip(interpolated_z_positions, interpolated_xy_positions):
#                                 info_file.write(f"{z:.2f}, {x:.2f}, {y:.2f}\n")

import os
import numpy as np
import nrrd
import matplotlib.pyplot as plt

input_folder = 'F://MargaridaP//CT_DICOM_extr_original'
outpath = 'F://MargaridaP//interpolated_SR'

def weighted_average_interpolation(slice1, slice2, distance1, distance2):
    interpolated_slice = (distance2 * slice1 + distance1 * slice2) / (distance1 + distance2)
    return interpolated_slice

def get_last_CTCA_image(folder_path):
    largest_x = -1
    largest_x_filename = None
    for filename in os.listdir(folder_path):
        if filename.startswith('CTCA_') and filename.endswith('.nrrd'):
            x = int(filename.split('_')[1].split('.')[0])
            if x > largest_x:
                largest_x = x
                largest_x_filename = filename
    return os.path.join(folder_path, largest_x_filename)

if not os.path.exists(outpath):
    os.makedirs(outpath)

for folder in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, folder)
    if os.path.isdir(folder_path):
        ct_filename = None
        ctca_filename = get_last_CTCA_image(folder_path)

        for filename in os.listdir(folder_path):
            if filename.startswith('CT_') and filename.endswith('.nrrd'):
                ct_filename = os.path.join(folder_path, filename)

            if ct_filename and ctca_filename:
                ct_data, ct_header = nrrd.read(ct_filename)
                ctca_data, ctca_header = nrrd.read(ctca_filename)
                
                num_slices_ct = ct_data.shape[2]
                space_directions_ct = ct_header['space directions']
                slice_thickness_ct = space_directions_ct[2, 2]

                num_slices_ctca = ctca_data.shape[2]
                space_directions_ctca = ctca_header['space directions']
                slice_thickness_ctca = space_directions_ctca[2, 2]

                z_positions_ct = [slice_thickness_ct * z for z in range(num_slices_ct)]
                z_positions_ctca = [slice_thickness_ctca * z for z in range(num_slices_ctca)]

                for i in range(num_slices_ct - 1):
                    slice1 = ct_data[:, :, i]
                    slice2 = ct_data[:, :, i + 1]
                    z1 = z_positions_ct[i]
                    z2 = z_positions_ct[i + 1]

                    interpolated_slices = []
                    interpolated_z_positions = []

                    for z_ctca in z_positions_ctca:
                        if z1 < z_ctca < z2:
                            distance_to_slice2 = z_ctca - z1
                            distance_to_slice1 = z2 - z_ctca
                            interpolated_slice = weighted_average_interpolation(slice1, slice2, distance_to_slice1, distance_to_slice2)
                            interpolated_slices.append(interpolated_slice)
                            interpolated_z_positions.append(z_ctca)
                    
                    if interpolated_slices:
                        patient_id = folder
                        base_name = os.path.basename(ct_filename)

                        output_base_name = f'{patient_id}_{base_name}_{i}'
                        np.save(os.path.join(outpath, f'{output_base_name}_reconstructed.npy'), np.array(interpolated_slices))
                        with open(os.path.join(outpath, f'{output_base_name}_z_positions.txt'), 'w') as z_file:
                            z_file.write(f'{z1}\n{z2}\n')
                            for z in interpolated_z_positions:
                                z_file.write(f'{z}\n')
