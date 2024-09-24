import os

mainpath = os.path.split(os.path.split(os.path.realpath(__file__))[0])[0]
mainpath = os.path.join(mainpath, 'data')
datapaths = {'cardiac_ct': os.path.join(mainpath, 'CT_DICOM_extr'),
             'cardiac_fat': os.path.join(mainpath, 'CardiacFat'),
             'osic': os.path.join(mainpath, 'OSIC'),
             'epicheart': os.path.join(mainpath, 'EPICHEART')}

csvfpaths = {}
for k, f in datapaths.items():
    csvfpaths[k] = os.path.join(f, 'metadata.csv')
respath = os.path.join(mainpath, 'results')
