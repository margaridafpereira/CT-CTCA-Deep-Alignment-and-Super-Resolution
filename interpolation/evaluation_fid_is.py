from fid_score import calculate_fid_given_paths
import torch

true_slice_path = "F:\\MargaridaP\\real_slices_png"
interpolated_imageSR_path = "F:\\MargaridaP\\unet_slices_png"

fid_score = calculate_fid_given_paths([true_slice_path, interpolated_imageSR_path], batch_size=10, cuda=False, dims=2048)
print("FID Score:", fid_score)

