import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from scipy import interp
def SaveTensorImage(image, path):
    image = Image.fromarray((np.array(image.cpu()) * 255).astype(np.uint8)).convert("L")
    if path is None:
        image.show()
    else:
        image.save(path)

def multi_AUC(predict, label, needThreshold = False,n_classes=3):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], threshold = metrics.roc_curve(label[:, i], predict[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(label.ravel(), predict.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] =metrics.auc(fpr["macro"], tpr["macro"])
    return fpr["macro"],tpr["macro"], roc_auc["macro"]

def CalculateAUC(predict, label, needThreshold = False,multi=True,multi_len = 4):
    if multi==True:
        _,n_classes = predict.shape
        if multi_len==4:
            labels = [0, 1, 2, 3]
        else:
            labels = [0, 1, 2, 3,4]
        ytrue = label_binarize(label, classes=labels)
        ypreds = predict
        fpr, tpr, auc = multi_AUC(ypreds, ytrue, needThreshold,n_classes)
    else:
        fpr, tpr, threshold = metrics.roc_curve(label, predict, drop_intermediate=True)  # drop_intermediate=False
        auc = metrics.auc(fpr, tpr)
    if needThreshold ==False:
        return auc, fpr, tpr
    else:
        return auc, fpr, tpr,threshold


def BinaryClassificationMetric(output, label, TP, FP, TN, FN, index):
    output = nn.functional.softmax(output, dim = 1)
    _, predict = torch.max(output, 1)
    P = (predict.int() == 1).int()
    N = (predict.int() == 0).int()
    TP[index] += (P * label.int()).sum().item()
    FP[index] += (P * (1 - label.int())).sum().item()
    TN[index] += (N * (1 - label.int())).sum().item()
    FN[index] += (N * label.int()).sum().item()


def ClassificationMetrics(TP, FP, TN, FN, epsilon = 0):
    accuracy = (TP + TN) / (TP + FP + TN + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    specificity = TN / (TN + FP + epsilon)
    sensitivity = TP / (TP + FN+ epsilon)
    PPV = TP / (TP + FP + epsilon)
    NPV = TN / (TN + FN + epsilon)
    F1 = 2*TP/(2*TP+FP+FN+epsilon)
    return accuracy, recall, precision, specificity, sensitivity, PPV, NPV, F1
