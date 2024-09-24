import os, sys
import csv
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from skimage.measure import label
from skimage import morphology
from sklearn.utils.multiclass import type_of_target
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import utils_csv

sys.path.append('..')
import CXR
from utils_data import pil_loader
from evalfuncs import plotPRC, plotROC

def getModelPredictions(fpath):
    tgt, pred = [],[]
    pred_name = []
    for fname in os.listdir(fpath):
        if fname.split('_')[-1] == 'reconstructed.png':
            fname_target = fname.replace('_reconstructed.','_target.')
            if os.path.isfile(os.path.join(fpath,fname_target)):
                pred.append(np.array(pil_loader(os.path.join(fpath,fname))))
                tgt.append(np.array(pil_loader(os.path.join(fpath,fname_target))))
                pred_name.append(fname)
            else:
                print(f'No target found for {fname}.')

    print(f'{len(tgt)} CXR masks found for {fpath}.')
    return tgt,pred,pred_name

def getConnCompLungs(pred):
    predPostProcessed = []
    for p in pred:
        #fig, (ax0, ax1) = plt.subplots(1, 2)
        #ax0.imshow(p)

        p = np.round(p/np.max(p))
        selem = morphology.star(int(np.max(np.shape(p))/200))
        p = morphology.binary_closing(p, selem)
        p = morphology.binary_opening(p, selem)

        plabel = label(p)
        labelu,counts = np.unique(plabel,return_counts=True)
        maxcounts = np.max(counts[1:])
        for l,c in zip(labelu[1:],counts[1:]):
            if c/maxcounts < 1/3:
                p[plabel==l] = 0

        #ax1.imshow(p)
        #plt.show()
        predPostProcessed.append(p)

    return predPostProcessed


def calcIoU(t, p):
    return np.sum(np.logical_and(t,p))/np.sum(np.logical_or(t,p))


def calcDiceScore(t, p):
    return 2 * np.sum(np.logical_and(t,p))/(np.sum(t) + np.sum(p))


def getDistMet(tgt,pred, func):
    distMet = []
    for t,p in zip(tgt,pred):
        t = np.round(t/255)
        p = np.round(p/np.max(p))
        distMet.append(func(t,p))
    return distMet


def calcMSE(img1, img2):
    # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])

    # return the MSE, the lower the error, the more "similar"the two images are
    return err

def calcSSIM(img1, img2):
    return ssim(img1, img2)

def createTrainValLossGraphs(modelPath):
    fold_names = ['fold1history', 'fold2history', 'fold3history', 'fold4history', 'fold5history']

    for fold_name in fold_names:
        name_csv = modelPath + '\\' + fold_name + '.csv'

        history_csv = pd.read_csv(name_csv)

        training_loss = history_csv['train_loss']
        validation_loss = history_csv['valid_loss']

        training_accuracy = history_csv['train_acc']
        validation_accuracy = history_csv['valid_acc']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        ax1.plot(training_loss, 'r--')
        ax1.plot(validation_loss, 'b-')
        ax1.legend(['Training Loss', 'Validation Loss'])

        ax2.plot(training_accuracy, 'r--')
        ax2.plot(validation_accuracy, 'b-')
        ax2.legend(['Training Sensitivity', 'Validation Sensitivity'])

        plt.title('LossAcc ' + fold_name)
        graphName = dset + 'lossAcc' + fold_name
        resultsGraphsFolder = os.path.join(model_folder[0], 'results_graphs')
        plt.savefig(os.path.join(resultsGraphsFolder, graphName))
        plt.show()

        return [training_loss, validation_loss, training_accuracy, validation_accuracy]

if __name__ == '__main__':
    model_folder = [os.path.join(CXR.respath, 'unet_JSRT_SingleClassHeart300epochs_lr10e4_512_5blocks')]

    dset = 'JSRT'
    subset = 'test'
    dset = dset+'_'+subset

    meanDiceScore = []
    meanSTD = []

    meanDiceScorePP = []
    meanSTD_PP = []

    allDiceScoresOrdered = []

    k_folds = [1, 2, 3, 4, 5]
    for k in k_folds:
        results_folder = [os.path.join(mfolder, dset) for mfolder in model_folder]
        res_fpath = os.path.join(results_folder[0], 'fold{}predictions'.format(k))
        tgt, pred, pred_name = getModelPredictions(res_fpath)

        print('Dice Score Before Post-Processing for Fold {}:'.format(k))
        diceScoreList = getDistMet(tgt, pred, calcDiceScore)

        orderedPredNames = [x for _, x in sorted(zip(diceScoreList, pred_name))]
        orderedDiceScoreList = sorted(diceScoreList)
        allDiceScoresOrdered.append(orderedDiceScoreList)
        allDiceScoresOrdered.append(orderedPredNames)

        meanDiceScore_fold = np.mean(diceScoreList)
        diceScoreSTD_fold = np.std(diceScoreList)
        meanDiceScore.append(meanDiceScore_fold)
        meanSTD.append(diceScoreSTD_fold)

        print(meanDiceScore_fold)
        print(diceScoreSTD_fold)

        # The Post-Processing step of the predictions is to get the two biggest connected components of the prediction.
        pred = getConnCompLungs(pred)

        print('Dice Score After Post-Processing for Fold {}:'.format(k))
        diceScoreListPP = getDistMet(tgt, pred, calcDiceScore)

        meanDiceScore_fold_PP = np.mean(diceScoreListPP)
        diceScoreSTD_fold_PP = np.std(diceScoreListPP)
        meanDiceScorePP.append(meanDiceScore_fold_PP)
        meanSTD_PP.append(diceScoreSTD_fold_PP)

        print(meanDiceScore_fold_PP)
        print(diceScoreSTD_fold_PP)

    totalMeanDiceScore = np.mean(meanDiceScore_fold)
    totalMeanSTDDiceScore = np.mean(diceScoreSTD_fold)
    print("\n Mean Dice Score of all folds: ", totalMeanDiceScore)
    print("\n Std Dice Score of all folds : ", totalMeanSTDDiceScore)

    totalMeanDiceScorePP = np.mean(meanDiceScore_fold_PP)
    totalMeanSTDDiceScorePP = np.mean(diceScoreSTD_fold_PP)
    print("\n Mean Dice Score of all folds post processed: ", totalMeanDiceScorePP)
    print("\n Std Dice Score of all folds post processed: ", totalMeanSTDDiceScorePP)

    results = pd.Series([totalMeanDiceScore, totalMeanSTDDiceScore, totalMeanDiceScorePP, totalMeanSTDDiceScorePP])
    results.to_csv(os.path.join(model_folder[0], '{}results.csv'.format(dset)), index=False)

    header = ['fold1_dsc', 'fold1_fname', 'fold2_dsc', 'fold2_fname', 'fold3_dsc', 'fold3_fname',
              'fold4_dsc', 'fold4_fname', 'fold5_dsc', 'fold5_fname']
    allDiceScoresOrdered = zip(*allDiceScoresOrdered)
    utils_csv.writeCsvOrderedPreds(os.path.join(model_folder[0], '{}_orderedPreds.csv'.format(dset)), allDiceScoresOrdered, header=header)

    # # PLot Data
    # plt.figure(figsize=(15, 10))
    # data = plt.boxplot(widths=0.50, positions=[0, 1, 2, 3, 4, 5],
    #                    labels=['{} Dice Score', '{} Dice Score PP', 'DL Montgomery', 'FL Montgomery', 'DL Shenzhen', 'FL Shenzhen'],
    #                    x=[jsrt, jrst_tversky_dsc, montgomery, montgomery_tversky_dsc, shenzen_dsc, shenzen_tversky_dsc])
    # plt.title('Dice Scores in the Datasets', fontsize=30)
    # plt.xticks(size=17)
    # plt.yticks(size=25)
    # plt.show()

    # for t,p in zip(tgt, pred):
    #     fig, (ax0, ax1) = plt.subplots(1, 2)
    #     ax0.imshow(t)
    #     ax1.imshow(p)
    #     plt.show()

    # Creates the graph with the train and validation loss and accuracy across the epochs for each fold
    #createTrainValLossGraphs(model_folder[0])