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

sys.path.append('..')
import CXR
from utils_data import pil_loader
from evalfuncs import plotPRC, plotROC

import utils_csv

def getModelPredictions(fpath, class_name=''):
    tgt, pred = [],[]
    pred_name = []
    for fname in os.listdir(fpath):
        if fname.split('_')[-1] == '{}.png'.format(class_name) and fname.split('_')[-2] == 'mask':
            fname_target = fname.replace('_mask_{}.'.format(class_name),'_target_{}.'.format(class_name))
            if os.path.isfile(os.path.join(fpath,fname_target)):
                target = np.array(pil_loader(os.path.join(fpath,fname_target)))
                #if target.all() != np.zeros((512,512)).all():
                pred.append(np.array(pil_loader(os.path.join(fpath,fname))))
                tgt.append(target)
                pred_name.append(fname)
                # else:
                #     print(f'No target found for {fname}.')
            else:
                print(f'No target found for {fname}.')

    print(f'{len(tgt)} CXR masks found for {class_name} {fpath}.')
    return tgt, pred, pred_name

def getModelPredictionsMulticlass(fpath, class_names=['lungs', 'heart', 'clavicles']):
    tgt, pred = [[], [], []], [[], [], []]
    for i, class_name in enumerate(class_names):
        tgt[i], pred[i], pred_name = getModelPredictions(fpath, class_name)
    return tgt, pred, pred_name

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
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img1.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
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
    model_folder = [os.path.join(CXR.respath, 'JSRT_Mnt_Multiclass_FixedWeights')]

    dset = 'Montgomery'
    subset = 'test'
    dset = dset+'_'+subset

    meanDiceScore = [[], [], []]
    meanSTD = [[], [], []]

    meanDiceScorePP = [[], [], []]
    meanSTD_PP = [[], [], []]
    allDiceScoresOrderedLungs = []
    allDiceScoresOrderedHeart = []
    allDiceScoresOrderedClavicles = []

    k_folds = [1, 2, 3, 4, 5]
    class_names = ['lungs', 'heart', 'clavicles']
    for k in k_folds:
        results_folder = [os.path.join(mfolder, dset) for mfolder in model_folder]
        res_fpath = os.path.join(results_folder[0], 'fold{}predictions'.format(k))
        tgt, pred, pred_name = getModelPredictionsMulticlass(res_fpath)

        for i, class_name in enumerate(class_names):
            print('Dice Score Before Post-Processing for Fold {}. Class:{}'.format(k, class_name))
            diceScoreList = getDistMet(tgt[i], pred[i], calcDiceScore)

            orderedPredNames = [x for _, x in sorted(zip(diceScoreList, pred_name))]
            orderedDiceScoreList = sorted(diceScoreList)

            if class_name == 'lungs':
                allDiceScoresOrderedLungs.append(orderedDiceScoreList)
                allDiceScoresOrderedLungs.append(orderedPredNames)
            elif class_name == 'heart':
                allDiceScoresOrderedHeart.append(orderedDiceScoreList)
                allDiceScoresOrderedHeart.append(orderedPredNames)
            elif class_name == 'clavicles':
                allDiceScoresOrderedClavicles.append(orderedDiceScoreList)
                allDiceScoresOrderedClavicles.append(orderedPredNames)


            meanDiceScore_fold = np.mean(diceScoreList)
            diceScoreSTD_fold = np.std(diceScoreList)
            meanDiceScore[i].append(meanDiceScore_fold)
            meanSTD[i].append(diceScoreSTD_fold)

            print(meanDiceScore_fold)
            print(diceScoreSTD_fold)

            # The Post-Processing step of the predictions is to get the two biggest connected components of the prediction.
            pred[i] = getConnCompLungs(pred[i])

            print('Dice Score After Post-Processing for Fold {}. Class:{}'.format(k, class_name))
            diceScoreListPP = getDistMet(tgt[i], pred[i], calcDiceScore)

            meanDiceScore_fold_PP = np.mean(diceScoreListPP)
            diceScoreSTD_fold_PP = np.std(diceScoreListPP)
            meanDiceScorePP[i].append(meanDiceScore_fold_PP)
            meanSTD_PP[i].append(diceScoreSTD_fold_PP)

            print(meanDiceScore_fold_PP)
            print(diceScoreSTD_fold_PP)


    resultsLines = []
    for i, class_name in enumerate(class_names):
        totalMeanDice = np.mean(meanDiceScore[i])
        totalMeanDiceSTD= np.mean(meanSTD[i])
        print("\n Class:{} Mean Dice Score of all folds:{}".format(class_name, totalMeanDice))
        print("\n Class:{} Std Dice Score of all folds :{}".format(class_name, totalMeanDiceSTD))

        totalMeanDicePP = np.mean(meanDiceScorePP[i])
        totalMeanDiceSTDPP = np.mean(meanSTD_PP[i])
        print("\n Class:{} Mean Dice Score of all folds post processed:{}".format(class_name, totalMeanDicePP))
        print("\n Class:{} Std Dice Score of all folds post processed:{}".format(class_name, totalMeanDiceSTDPP))

        resultsLines.append([totalMeanDice, totalMeanDiceSTD, totalMeanDicePP, totalMeanDiceSTDPP])


    header = ['MeanDice', 'MeanDiceSTD', 'MeanDicePP', 'MeanDiceSTDPP']
    utils_csv.writeCsv(os.path.join(model_folder[0], '{}results.csv'.format(dset)), resultsLines, header=header)

    header = ['fold1_dsc', 'fold1_fname', 'fold2_dsc', 'fold2_fname', 'fold3_dsc', 'fold3_fname',
              'fold4_dsc', 'fold4_fname', 'fold5_dsc', 'fold5_fname']

    allDiceScoresOrderedLungs = zip(*allDiceScoresOrderedLungs)
    allDiceScoresOrderedHeart = zip(*allDiceScoresOrderedHeart)
    allDiceScoresOrderedClavicles = zip(*allDiceScoresOrderedClavicles)

    utils_csv.writeCsvOrderedPreds(os.path.join(model_folder[0], '{}_orderedPredsLungs.csv'.format(dset)),
                                   allDiceScoresOrderedLungs, header=header)
    utils_csv.writeCsvOrderedPreds(os.path.join(model_folder[0], '{}_orderedPredsHeart.csv'.format(dset)),
                                   allDiceScoresOrderedHeart, header=header)
    utils_csv.writeCsvOrderedPreds(os.path.join(model_folder[0], '{}_orderedPredsClavicles.csv'.format(dset)),
                                   allDiceScoresOrderedClavicles, header=header)

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
