import re
from scipy import interpolate
import os, sys
import numpy as np
import nrrd
sys.path.insert(0, '..') #main path
import utils.utils_paths
import matplotlib.pyplot as plt


def get_space_meshgrid(header):
    xv, yv, zv = np.meshgrid(np.linspace(0, header['sizes'][0] - 1, header['sizes'][0]),
                              np.linspace(0, header['sizes'][1] - 1, header['sizes'][1]),
                              np.linspace(0, header['sizes'][2] - 1, header['sizes'][2]))

    xyz_coord = np.concatenate((xv.flatten()[None, :], yv.flatten()[None, :], zv.flatten()[None, :]), axis=0)
    xyz_space = np.matmul(header['space directions'], xyz_coord)
    xyz_space[0, :] += header['space origin'][0]
    xyz_space[1, :] += header['space origin'][1]
    xyz_space[2, :] += header['space origin'][2]

    return xyz_space

def interpolate_and_align_CTCA(ctca_slices, ctca_header, ct_header):
    # Find spatial boundaries of CT
    xmin = min(np.reshape(get_space_meshgrid(ct_header)[0, :], ct_header['sizes'])[0, :, 0])
    ymin = min(np.reshape(get_space_meshgrid(ct_header)[1, :], ct_header['sizes'])[:, 0, 0])
    xmax = max(np.reshape(get_space_meshgrid(ct_header)[0, :], ct_header['sizes'])[0, :, 0])
    ymax = max(np.reshape(get_space_meshgrid(ct_header)[1, :], ct_header['sizes'])[:, 0, 0])

    # Define space for interpolation
    x_new = np.linspace(xmin, xmax, ct_header['sizes'][0])
    y_new = np.linspace(ymin, ymax, ct_header['sizes'][1])

    aligned_ctca_slices = []

    for i, ctca_slice in enumerate(ctca_slices):
        # Get the corresponding CTCA slice index
        z_ind_ctca = i  # Adjust if needed

        # Make interpolation function
        f_ctca = interpolate.RectBivariateSpline(
            np.reshape(get_space_meshgrid(ctca_header)[0, :], ctca_header['sizes'])[0, :, z_ind_ctca],
            np.reshape(get_space_meshgrid(ctca_header)[1, :], ctca_header['sizes'])[:, 0, z_ind_ctca],
            ctca_slice)

        # Interpolate the new image
        ctca_new = f_ctca(x_new, y_new)

        # Handle values outside the bounds
        out_of_bounds_mask_x = (x_new < np.min(get_space_meshgrid(ctca_header)[0, :])) | (x_new > np.max(get_space_meshgrid(ctca_header)[0, :]))
        out_of_bounds_mask_y = (y_new < np.min(get_space_meshgrid(ctca_header)[1, :])) | (y_new > np.max(get_space_meshgrid(ctca_header)[1, :]))

        ctca_new[out_of_bounds_mask_x, :] = np.NaN
        ctca_new[:, out_of_bounds_mask_y] = np.NaN

        aligned_ctca_slices.append(ctca_new)

    return aligned_ctca_slices
        
dataname = 'cardiac_ct'
#inpath= "D://MargaridaP//pericardial_segmentation//data//falta"
inpath = os.path.join(utils.utils_paths.datapaths[dataname]+'_original')
outpath = os.path.join(utils.utils_paths.datapaths[dataname])

if not os.path.isdir(outpath):
    os.makedirs(outpath)
    os.makedirs(os.path.join(outpath, 'CT2'))
    os.makedirs(os.path.join(outpath, 'CTCA_aligned1'))
    os.makedirs(os.path.join(outpath, 'CT_pos'))
    os.makedirs(os.path.join(outpath, 'CTCA_aligned_pos'))

for f in os.listdir(inpath):
    patient_path = os.path.join(inpath, f)

    if os.path.isdir(patient_path):
        ct_files = [file for file in os.listdir(patient_path) if file.startswith('CT_') and file.endswith('.nrrd')]
        ctca_files = [file for file in os.listdir(patient_path) if file.startswith('CTCA_') and file.endswith('.nrrd')]

        #ctca_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))

        for ctca_file in ctca_files:
            #last_ctca_file = os.path.join(patient_path, ctca_files[-1])
            print (patient_path)
            #print("CTCA File:", ctca_file)
            ctca_data, ctca_header = nrrd.read(os.path.join(patient_path, ctca_file))
            #ctca_data, ctca_header = nrrd.read(ctca_file)
            
            start_z = ctca_header['space origin'][2]  
            slice_thickness = ctca_header['space directions'][2][2]  
            num_slices_ctca = ctca_header['sizes'][2]  

            initial_slice_position = start_z
            final_slice_position = start_z + (num_slices_ctca - 1) * slice_thickness

            #print (initial_slice_position, final_slice_position)
            
            for ct_file in ct_files:
                ct_slices_within_region = []
                ct_positions_within_region = []
                
                if ct_file.endswith('.nrrd'):
                    ct_file_path = os.path.join(patient_path, ct_file)
                    ct_data, ct_header = nrrd.read(ct_file_path)
            
                    ct_positions_z = ct_header['space origin'][2]
                    ct_slice_thickness = ct_header['space directions'][2][2]
                    num_slices_ct = ct_header['sizes'][2]
                    
                    if ct_slice_thickness<= slice_thickness:
                        break

                    else:
                        for i in range(num_slices_ct):
                            current_slice_position = ct_positions_z + i * ct_slice_thickness
    
                            if initial_slice_position - ct_slice_thickness  <= current_slice_position <= final_slice_position + ct_slice_thickness :
                                ct_slices_within_region.append(ct_data[:, :, i])
                                ct_positions_within_region.append(current_slice_position)
                             
                        #print (ct_positions_within_region)
                        
                        for i in range(0, len(ct_slices_within_region) - 1, 1):
                            ct_slice_1 = ct_slices_within_region[i]
                            ct_slice_2 = ct_slices_within_region[i + 1]
            
                            ct_data = [ct_slice_1, ct_slice_2]
                            np.save(os.path.join(outpath, 'CT2', f'{f}_{ct_file}&{ctca_file}_{i}.npy'), ct_data)
            
                            pos_1 = ct_positions_within_region[i]
                            pos_2 = ct_positions_within_region[i + 1]
                                                
                            file_path = os.path.join(outpath, 'CT_pos',f'{f}_{ct_file}&{ctca_file}_{i}.txt')
                            
                           
    
                            ctca_slices = []
                            ctca_slices_positions = []
                            
                            for j in range(num_slices_ctca):
                                current_ctca_slice_position = start_z + j * slice_thickness
            
                                if pos_1 - 0.5*ct_slice_thickness <= current_ctca_slice_position <= pos_2+ 0.5*ct_slice_thickness:
                                    ctca_slices.append(ctca_data[:, :, j])
                                    ctca_slices_positions.append(current_ctca_slice_position)
                                    #print("CTCA")
                                    #print(current_ctca_slice_position)
                                    #np.save(os.path.join(outpath, 'CTCA', f'{f}_{ct_file}_{i}.npy'), ctca_slices)
                                
                            aligned_slices = interpolate_and_align_CTCA(ctca_slices, ctca_header, ct_header)
                            np.save(os.path.join(outpath, 'CTCA_aligned1', f'{f}_{ct_file}&{ctca_file}_{i}.npy'), aligned_slices)
                    

                            file_path = os.path.join(outpath, 'CTCA_aligned_pos',f'{f}_{ct_file}&{ctca_file}_{i}.txt')
                            
                            # with open(file_path, 'w') as file:
                            #     for position in ctca_slices_positions:
                            #         file.write(f"{position}\n")
                                
                            # plt.figure(figsize=(15, 5))
                                    
                            # for i, slice_data in enumerate(aligned_slices):
                            #             plt.subplot(1, len(aligned_slices), i+1)
                            #             plt.imshow(slice_data, cmap='gray')
                            #             plt.title(f'Aligned CTCA Slice {i+1}')
                            #             plt.axis('off')
    
                            #             plt.show()