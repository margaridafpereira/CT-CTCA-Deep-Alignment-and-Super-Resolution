import os, sys
import numpy as np
from PIL import Image
import pandas as pd

sys.path.insert(0, '..')
import utils.utils_paths

dataname = 'cardiac_ct'
path = 'F://MargaridaP' #os.path.join(utils.utils_paths.datapaths[dataname])

datafolder = 'Data'
targetfolder = 'Target'
ctfolder = 'CT_pos'
ctcafolder = 'CTCA_pos'

header = ['Data', 'Target', 'CT_pos', 'CTCA_pos', 'Fold']
lines = []
npats_fold = 40
npats = 0
pat = ''
fold = 0

for f in os.listdir(os.path.join(path, datafolder)):
    if not f.split('_')[0] == pat:
        pat = f.split('_')[0]
        npats += 1
        if npats > 40:
            npats = 1
            fold += 1
        print(pat, fold, npats)
        
    lines.append([os.path.join(datafolder, f), os.path.join(targetfolder, f), os.path.join(ctfolder, f).replace('.npy', '.txt'), os.path.join(ctcafolder, f).replace('.npy', '.txt'), fold])

df = pd.DataFrame(lines, columns=header)
df.to_csv(os.path.join(path, 'cardiac_ct_unet1.csv'), index=False)



