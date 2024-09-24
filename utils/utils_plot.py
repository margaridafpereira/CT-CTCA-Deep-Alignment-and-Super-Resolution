from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
cmap = cm.get_cmap('tab20')
lstyle = ['-','--','-.',':']
mstyle = ['o','v','^','<','>','s','P','*']
import os
from PIL import Image
import numpy as np
import random

import CXR
from utils_data import pil_loader

def getImage(imgname,folder = None):
    if folder is None:
        img = None
        for k,ffolder in CXR.datapaths.items():
            print(ffolder)
            try:
                img = getImage(imgname,os.path.join(ffolder,'imgs'))
                break
            except:
                pass
        if img is None:
            print(imgname)
            img = getImage(os.path.split(imgname)[-1], os.path.sep.join(os.path.split(imgname)[:-1]))
    else:
        imgfpath = os.path.join(folder,imgname)
        print(imgfpath)
        img = pil_loader(imgfpath)
    return img

def plotImage(lines,header,imgname, img = None, iclass = None):
    print(imgname)
    imgind = header.index('filename')
    annind = header.index('class_id')
    x0ind = header.index('x_min')
    x1ind = header.index('x_max')
    y0ind = header.index('y_min')
    y1ind = header.index('y_max')
    radind = header.index('rad_id')
    try:
        probind = header.index('bbox_prob')
        prediction = True
    except:
        prediction = False

    if iclass is None:
        iclass = [i for i in range(len(CXR.datapaths))]
    elif not isinstance(iclass,list):
        iclass = [iclass]

    if img is None:
        img = getImage(imgname)
        
    imglist = np.array([l[imgind] for l in lines])
    ind = np.where(imglist == imgname)[0]
    
    plt.figure()
    plt.imshow(img,'gray')
    annlist = []
    radu = np.unique([l[radind] for l in [lines[i] for i in ind]])
    for i in ind:
        ann = int(lines[i][annind])
        if ann in iclass:
            if not ann in annlist:
                annlist.append(ann)
            if not ann == 14:
                x0 = float(lines[i][x0ind])
                x1 = float(lines[i][x1ind])
                y0 = float(lines[i][y0ind])
                y1 = float(lines[i][y1ind])
                cx = [x0,x0,x1,x1,x0]
                cy = [y0,y1,y1,y0,y0]
                plt.plot(cx,cy,color=cmap(ann),linestyle=lstyle[np.where(radu==lines[i][radind])[0][0]])
    plt.xticks([])
    plt.yticks([])

if __name__ == '__main__':
    import utils_csv
    datapath = CXR.datapaths['VinBigData']
    csvfpath = os.path.join(datapath,'filteredVinBigDataBBoxes.csv')
    header, lines = utils_csv.readCsv(csvfpath)
    
    fnameind = header.index('filename')
    classind = header.index('class_id')
    imglist = [l[0] for l in lines]
    random.shuffle(lines)
    plotImage(lines,header,imgname='000ae00eb3942d27e0b97903dd563a6e.png')
    plt.show()
