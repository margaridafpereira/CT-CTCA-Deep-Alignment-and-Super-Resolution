import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def sigmoid(x):
    return 1/(1+np.exp(-float(x)))

def get_confusion_matrix(labels0,labels1,nclass = None):
    # Builds confusion matrix from labelling by two readers
    # labels0 - List with labels (integers 0 to N) given by reader 0
    # labels1 - List with labels (integers 0 to N) given by reader 1
    # nclass (optional) - Number of classes to consider. Can be a single number N for a NxN confusion matrix or a [N0,N1] for a rectangular N0xN1 confusion matrix
    # returns confusion matrix as list of lists
    if nclass == None:
        nclass0 = max(labels0)
        nclass1 = max(labels1)
    elif isinstance(nclass,list):
        nclass0 = nclass[0]
        nclass1 = nclass[1]
    else:
        nclass0 = nclass
        nclass1 = nclass        
    
    confmat = [[0 for _ in range(nclass1+1)] for _ in range(nclass0+1)]
    for l0,l1 in zip(labels0,labels1):
        if l0>=0 and l1>=0:
            confmat[l0][l1] += 1
    return confmat

def plot_confusion_matrix(data, xylabels=None, xyaxislabels = None, title='',
                          txtlabels=None, strfmt = 'flt', txtlabelsVal = True, cmap=plt.cm.Blues, clim = None):
    # Plots confusion matrix as given by get_confusion_matrix
    # data - confusion matrix as given by get_confusion_matrix
    # xylabels (optional) - Labels for each class to place on x and y axis. Should be a list of lists [labelsx,labelsy] with labels for both axis.
    # xyaxislabels (optional) - Labels for each axis (e.g. name of readers). Should be a list [labelx,labely] with a string for each axis.
    # title (optional) - Figure title. Should be a string.
    # txtlabels (optional) - Gives additional/replacement text to show on each confusion matrix cell. Should be a list of lists with dimensions equal to data
    # strfmt (optional) - Controls data format for text. 'flt' show two decimal digits, otherwise rounds to integer.
    # txtlabelsVal (optional) - Boolean to show/hide values on data for each cell as text.
    # cmap (optional) - Colormap to use for coloring cells.
    # clim (optional) - Colormap limits. Should be a list (e.g. [0,1]) with minimum and maximum colormap limits.
    if isinstance(data,list):
        data = np.asarray(data)
    
    if not xylabels:
        xylabels = [[l for l in range(len(data[0]))]]*2
    
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots()
    im = ax.imshow(data, interpolation='nearest', cmap=cmap)
    
    if not clim==None:
        im.set_clim(clim[0],clim[1])
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(data.shape[1]),
           yticks=np.arange(data.shape[0]),
           xticklabels=xylabels[1], yticklabels=xylabels[0])
    
    # Move xticks to top
    ax.xaxis.tick_top()
    
    # Place XY axis labels
    if xyaxislabels:
        ax.set_xlabel(xyaxislabels[1])
        ax.xaxis.set_label_position('top') 
        ax.set_ylabel(xyaxislabels[0])
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="left")

    # Loop over data dimensions and create text annotations.
    if strfmt == 'flt':
        datastrfmt = '{:.2f}'
    else:
        datastrfmt = '{:d}'
    if txtlabels:
        if txtlabelsVal:
            txtstr = datastrfmt+'\n{}'
        else:
            txtstr = '{}'
    else:
        txtstr = datastrfmt
    thresh = data.max() / 2.
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if txtlabels and txtlabelsVal:
                ax.text(j, i, txtstr.format(data[i,j],txtlabels[i][j]),
                        ha="center", va="center",
                        color="white" if data[i,j] > thresh else "black")             
            elif not txtlabels and txtlabelsVal:
                ax.text(j, i, txtstr.format(data[i,j]),
                        ha="center", va="center",
                        color="white" if data[i,j] > thresh else "black")
            elif txtlabels and not txtlabelsVal:
                ax.text(j, i, txtstr.format(txtlabels[i][j]),
                        ha="center", va="center",
                        color="white" if data[i,j] > thresh else "black")
    
    if not title=='':
        plt.gca().text(data.shape[1]/2-.5,data.shape[0]-.0,title,ha='center',va='center',weight='bold')
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    #cax = divider.append_axes("bottom", size="5%", pad=0.1)
    #plt.colorbar(im, cax=cax, orientation="horizontal")
    
    fig.tight_layout()    
    
    return fig

def calcKappa(confm,weight = None,debug = False):
    # Computes kappa for a confusion matrix as given by get_confusion_matrix
    # confm - confusion matrix as given by get_confusion_matrix
    # weight (optional) - Whether to have a weighted kappa. 'FleissCohen' or 'FC' for Fleiss-Cohen kappa and 'CicchettiAllison' or 'CA' for Cicchetti-Allison kappa
    # debug (optional) - Boolean for debugging. Gives prints throughout calculations.
    nclass = len(confm)
    
    kW =  [[] for l in range(nclass)]
    for l1 in range(nclass):
        for l2 in range(nclass):
            if weight == 'FleissCohen' or weight == 'FC':
                w = 1-(l1-l2)**2/(0-nclass+1)**2
            elif weight == 'CicchettiAllison' or weight == 'CA':
                w = 1-abs(l1-l2)/abs(0-nclass+1)
            else:
                if l1==l2:
                    w = 1
                else:
                    w = 0
            kW[l1].append(w)
    
    sumC = list(np.sum(confm,0))
    sumR = list(np.sum(confm,1))
    n = sum(sumR)
    
    pO = 0
    pE = 0
    for l1 in range(nclass):
        for l2 in range(nclass):
            pO += kW[l1][l2] * confm[l1][l2] / n
            pE += kW[l1][l2] * (sumR[l2]/n * sumC[l1]/n)
    
    if n>0:
        if pE<1.0:
            k = (pO-pE)/(1-pE)
        else:
            k = 1.0
    else:
        k = 0
    
    if debug:
        print(kW)
        for c in confm:
            print(c)
        print(pO,pE)
        print(k)
    
    return k

def calcAgr(confm, acases = None, debug = False):
    # Computes agreement for a confusion matrix as given by get_confusion_matrix
    # confm - confusion matrix as given by get_confusion_matrix
    # acases (optional) - List of lists describing which combinations are considered as agreement. If None is given, acases reverts to [[l,l] for l in range(nclass)], i.e.
#agreement is given when classes are equal.
    # debug (optional) - Boolean for debugging. Gives prints throughout calculations.
    
    nclass = len(confm)
    
    if not acases:
        acases = [[l,l] for l in range(nclass)]
    
    na = 0
    n = 0
    for l1 in range(nclass):
        for l2 in range(nclass):
            n += confm[l1][l2]
            if [l1,l2] in acases:
                    na += confm[l1][l2]
    
    if n>0:
        a = na/n
    else:
        a = 0
    
    if debug:
        for c in confm:
            print(c)        
        print(a,na,n)
        
    return a

def plotROC(gtl,pdp,confinterval = True,labelstr = 'Model',textstr = None,newFig = True,save_name = None, scatter = False, linestyle ={}, markerstyle = {}, return_val = False):
    # gtl - List (or list of lists) with GT labels
    # pdp - List or list of lists with prediction probabilities
    # confinterval (optional) - if True averages ROC across list of lists and plots average and 95% confidence interval
    # labelstr (optional) - string with name to show in plot legend
    # textstr (optional) - string to show additional information on top right corner
    # newFig (optional) - whether to create a new figure or plot on previous
    # save_name (optional) - if given, saves current figure to file path given
    # scatter (optional) - instead of line, plots as separate scatter points
    # lline (optional) - last line to apply same color for scatter
    if newFig:
        plt.figure()
    
    if not isinstance(gtl[0],list):
        confinterval = False
        fpr, tpr, thr = roc_curve(gtl,pdp)
        fpr = [fpr]
        tpr = [tpr]
        thr = [thr]
        gtl = [gtl]
        unique_pdp = np.unique(pdp)
        if len(unique_pdp)==2 and unique_pdp[0] == 0 and unique_pdp[1] == 1:
            scatter = True
        elif len(unique_pdp)==1:
            scatter = True
    else:
        fpr,tpr,thr = [],[],[]
        for gtk,pdk in zip(gtl,pdp):
            fprk, tprk, thrk = roc_curve(gtk,pdk,drop_intermediate=True)
            fpr.append(fprk)
            tpr.append(tprk)
            thr.append(thrk)
        if len(fpr)==1 or len(tpr)==1:
            confinterval = False

    if return_val:
        return fpr, tpr, thr

    aucs,accs,sens,spec,thrs = [],[],[],[],[]
    for fprk,tprk,thrk in zip(fpr,tpr,thr):
        aucs.append(auc(fprk,tprk))
        thrind = np.argmin(((1-fprk)**2+(tprk)**2)**.5)
        accs.append((tprk[thrind]*gtl[0].count(1)+(1-fprk[thrind])*gtl[0].count(0))/len(gtl[0]))
        thrs.append(thrk[thrind])
        score = [((1-t)**2+f**2)**.5 for f,t in zip(fprk,tprk)]
        indscore = np.argmin(score)
        sens.append(tprk[indscore])
        spec.append(1-fprk[indscore])

    print(labelstr)
    print('AUC: {:.3f}+-{:.3f}'.format(np.mean(aucs),np.std(aucs)))
    print('Acc: {:.3f}+-{:.3f}'.format(np.mean(accs),np.std(accs)))
    print('Thrs:',thrs)
    print('Sens: {:.3f}+-{:.3f}'.format(np.mean(sens),np.std(sens)))
    print('Spec: {:.3f}+-{:.3f}'.format(np.mean(spec),np.std(spec)))
    
    if confinterval and not scatter:
        fpru = []
        for fprk in fpr:
            fpru.extend(fprk)
        fpru = np.unique(fpru)
        fprmean = []
        tprmean = []
        fprdev = []
        tprdevup = []
        tprdevdown = []
        tprs = [0]
        for fprv in fpru:
            fprmean.append(fprv)
            tprmean.append(np.mean(tprs))
            tprdevdown.append(np.mean(tprs)-1.96*np.std(tprs)/np.sqrt(len(fpr)))            
            tprs = []
            for k,(fprk,tprk) in enumerate(zip(fpr,tpr)):
                for i,(f,t) in enumerate(zip(fprk,tprk)):
                    if f>fprv:
                        tprs.append(tprk[i-1])
                        break
            if tprs:
                fprmean.append(fprv)
                tprmean.append(np.mean(tprs))
                tprdevup.append(np.mean(tprs)+1.96*np.std(tprs)/np.sqrt(len(fpr)))
            else:
                tprdevup.append(tprdevdown[-1])
        
        l = plt.plot(fprmean,tprmean,label=labelstr,**linestyle)
        plt.gca().fill_between(fpru,tprdevdown,tprdevup,
                               facecolor = l[0].get_color(), alpha = 0.25)#,label = labelstr+'CI 95%')
        fprk,tprk = fprmean,tprmean
    else:
        for k,(fprk,tprk) in enumerate(zip(fpr,tpr)):
            if scatter:
                l = plt.scatter(fprk[1:-1],tprk[1:-1],label=labelstr,**markerstyle,zorder=3)
            else:
                l = plt.plot(fprk,tprk,label=labelstr,**linestyle)
            
    plt.gca().set_aspect('equal','box')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.legend(loc="lower right",fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_name:
        if textstr:
            plt.text(.99,.98,textstr, horizontalalignment='right',verticalalignment='top',bbox=dict(facecolor='white'), fontsize = 14)
        plt.savefig(save_name, format='png', bbox_inches='tight')
    
    if not return_val:
        return l
    else:
        return fpr, tpr, 0

def plotPRC(gtl,pdp,confinterval = True,labelstr = 'Model',textstr = None,newFig = True,save_name = None, scatter = False, linestyle ={}, markerstyle = {}, return_val = False):
    # gtl - List (or list of lists) with GT labels
    # pdp - List or list of lists with prediction probabilities
    # confinterval (optional) - if True averages ROC across list of lists and plots average and 95% confidence interval
    # labelstr (optional) - string with name to show in plot legend
    # textstr (optional) - string to show additional information on top right corner
    # newFig (optional) - whether to create a new figure or plot on previous
    # save_name (optional) - if given, saves current figure to file path given
    # scatter (optional) - instead of line, plots as separate scatter points    
    if newFig:
        plt.figure()
    
    if not isinstance(gtl[0],list):
        confinterval = False
        prc, rec, _ = precision_recall_curve(gtl,pdp)
        prc = [prc]
        rec = [rec]
        no_skill = sum(gtl) / len(gtl)
        unique_pdp = np.unique(pdp)
        if len(unique_pdp)==2 and unique_pdp[0] == 0 and unique_pdp[1] == 1:
            scatter = True        
    else:
        prc,rec,no_skill = [],[],0
        for gtk,pdk in zip(gtl,pdp):
            prck, reck, _ = precision_recall_curve(gtk,pdk)
            prc.append(prck)
            rec.append(reck)            
            no_skill += sum(gtk) / len(gtk)
        no_skill /= len(gtl)

    if return_val:
        return rec,prc, no_skill

    if confinterval and not scatter:
        recu = []
        for reck in rec:
            recu.extend(reck)
        recu = np.unique(recu)
        recmean = [0]
        prcmean = [1]
        recdev = []
        prcdevup = [1]
        prcdevdown = [1]
        for recv in recu:
            prcmin = []
            prcmax = []
            for k,(reck,prck) in enumerate(zip(rec,prc)):
                if recv in reck:
                    prcc = prck[reck==recv]
                    prcmin.append(min(prcc))
                    prcmax.append(max(prcc))
                else:                
                    for i,(r,p) in enumerate(zip(reck[::-1],prck)):
                        if r>recv:
                            prcmin.append(prck[::-1][i-1])
                            prcmax.append(prck[::-1][i-1])
                            break
            
            if prcmin:                
                recmean.append(recv)
                prcmean.append(np.mean(prcmax))
                prcdevup.append(np.mean(prcmax)+1.96*np.std(prcmax)/np.sqrt(len(rec)))
                prcdevdown.append(np.mean(prcmax)-1.96*np.std(prcmax)/np.sqrt(len(rec)))
                recmean.append(recv)
                prcmean.append(np.mean(prcmin))
                prcdevup.append(np.mean(prcmin)+1.96*np.std(prcmin)/np.sqrt(len(rec)))
                prcdevdown.append(np.mean(prcmin)-1.96*np.std(prcmin)/np.sqrt(len(rec)))                
            else:
                pass
    
        l = plt.plot(recmean,prcmean,label=labelstr,**linestyle)
        plt.gca().fill_between(recmean,prcdevdown,prcdevup,
                               facecolor = l[0].get_color(), alpha = 0.25)#,label = labelstr+'CI 95%')
    else:
        for k,(reck,prck) in enumerate(zip(rec,prc)):
            if scatter:
                l = plt.scatter(reck[1:-1],prck[1:-1],label=labelstr,**markerstyle)
            else:
                l = plt.plot(reck,prck,label=labelstr,**linestyle)
            
    if confinterval or not scatter:
        plt.plot([0,1],[no_skill,no_skill],'--',color=l[-1].get_color())
    plt.gca().set_aspect('equal','box')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.legend(loc="lower right")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if save_name:
        if textstr:
            plt.text(0.01,.02,textstr, horizontalalignment='left',verticalalignment='bottom',bbox=dict(facecolor='white'), fontsize = 14)
        plt.savefig(save_name, format='png', bbox_inches='tight')
    
    if not return_val:
        return l
    else:
        return rec,prc, no_skill