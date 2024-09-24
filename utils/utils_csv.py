import csv
import re

import numpy as np


def readCsv(csvpath, header=True):
    with open(csvpath) as f:
        reader = csv.reader(f)
        lines = list(reader)

    if header:
        return lines[0], lines[1:]
    else:
        return lines


def writeCsv(csvpath, lines, header=None):
    with open(csvpath, mode='w', newline='') as f:
        writer = csv.writer(f)

        if not header == None:
            writer.writerow(header)
        writer.writerows(lines)


def writeCsvOrderedPreds(csvpath, allDiceScoreOrdered, header=None):
    with open(csvpath, mode='w', newline='') as f:
        writer = csv.writer(f)

        if not header == None:
            writer.writerow(header)

        # allDiceScoreOrdered = np.array(allDiceScoreOrdered)
        # allDiceScoreOrdered = allDiceScoreOrdered.T
        # allDiceScoreOrdered = allDiceScoreOrdered.tolist()

        writer.writerows(allDiceScoreOrdered)


def readCsvDataLists(csvpath, targetstr=['filename', 'class']):
    header, lines = readCsv(csvpath)
    if isinstance(targetstr, str):
        targetstr = [targetstr]
    targetind = [header.index(t) for t in targetstr]
    datalists = [[] for _ in targetstr]
    for l in lines:
        for i, t in enumerate(targetind):
            datalists[i].append(l[t])

    if len(targetstr) == 1:
        return datalists[0]
    else:
        return datalists


def readGTCsv(csvpath, majority=True, tievote=1):
    imlist, tgtlist = readCsvDataLists(csvpath, targetstr=['relative_path', 'group_label'])
    if not majority:
        return imlist, tgtlist
    else:
        uim = []
        mtgt = []
        for im in imlist:
            if not im in uim:
                cl = [int(t) for t, i in zip(tgtlist, imlist) if i == im]
                cl = cl + [tievote]
                cl = max(set(cl), key=cl.count)
                uim.append(im)
                mtgt.append(cl)
        return uim, mtgt


def readPredCsv(csvpath):
    header, lines = readCsv(csvpath)
    datalists = []
    datalists.append([l[header.index('filename')] for l in lines])

    lbl = None
    for ind, h in enumerate(header):
        matchc = re.search('class(\d+)prob', h)
        try:
            cl = matchc.group(1)
        except:
            continue
        lbl_h = h.replace('class{}prob'.format(cl), '')
        if not lbl == lbl_h:
            if not lbl == None:
                datalists.append(dlist)
            dlist = []
            lbl = lbl_h
        dlist.append([l[ind] for l in lines])
    if not lbl == None:
        datalists.append(dlist)

    return datalists


'''
This function reads the bounding box csv file in order to extract only the filename
and the x and y coordinates of the bounding box location.
'''


def readBboxCsv(csvpath):
    datalists = readCsvDataLists(csvpath, ['filename', 'class_name', 'class_id', 'x_min', 'y_min', 'x_max', 'y_max',
                                           'bbox_prob'])

    return datalists
