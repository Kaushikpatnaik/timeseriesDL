import numpy as np
import pandas as pd
from sklearn.metrics import *
import matplotlib.pyplot as plt
import logging

module_logger = logging.getLogger('timeSeriesDL.compute_statistics')

def costTrainPlot(cost, logdir, custom_txt):

    plt.plot(np.linspace(1, len(cost), len(cost)), cost)
    plt.title('Cost per batch over the training run')
    plt.xlabel('# batch')
    plt.ylabel('avg. cost per batch')
    plt.savefig(logdir + custom_txt + '_traincost.png')

def compConfusion(op_prob,y_onehot):
    '''
    Return the confusion matrix based on op_probability and true labels
    Args:
        op_prob: num_samples x num_classes
        y_onehot: one-hot encoding of num_samples x num_classes

    Returns:

    '''

    y_predict = np.argmax(op_prob,axis=1)
    y_true = np.argmax(y_onehot,axis=1)

    confMatrix = classification_report(y_true,y_predict)
    return confMatrix

def compMetrics(op_prob, y_onehot):
    '''
    Returns the pr,roc,f1 and accuracy scores
    Note: Only works for binary classification
    Args:
        op_prob: num_samples x num_classes output probability
        y_onehot: one-hot encoding of true labels
        logdir
    Returns: Tuple of values
    '''

    y_true = np.argmax(y_onehot, axis=1)
    y_pred = np.argmax(op_prob,axis=1)

    auc_score = roc_auc_score(y_true, op_prob[:, 1])
    pr_score = average_precision_score(y_true, op_prob[:, 1])
    f_score = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    module_logger.info("PR AUC, ROC_AUC, F1_SCORE, Accuracy")
    module_logger.info((pr_score,auc_score,f_score,acc))

    return (pr_score,auc_score,f_score,acc)

def rocPrAucPlot(op_prob,y_onehot,logdir,folder):
    '''
    Plot the roc, PR Curve and return the AUC metric
    Note: Only works for binary classification
    Args:
        op_prob: num_samples x num_classes output probability
        y_onehot: one-hot encoding of true labels
        model: args.model
        folder: train, val or test

    Returns:

    '''

    y_true = np.argmax(y_onehot,axis=1)

    auc_score = roc_auc_score(y_true,op_prob[:,1])
    pr_score = average_precision_score(y_true,op_prob[:,1])

    # Plotted curves for the failure class only
    prec1, recall1, thres1 = precision_recall_curve(y_true,op_prob[:,1])
    prec2, recall2, thres2 = roc_curve(y_true,op_prob[:,1])

    fig, axes = plt.subplots(nrows=1,ncols=2)
    axes[0].plot(prec1,recall1)
    axes[1].plot(prec2,recall2)
    axes[1].set_title('ROC Curve '+str(auc_score))
    axes[0].set_title('PR Curve '+str(pr_score))
    plt.savefig(logdir+folder+'_roc_auc_curves.png')

    return auc_score, pr_score