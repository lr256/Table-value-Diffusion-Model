import os
import csv
import torch
import numpy as np
import matplotlib.pylab as plot
import sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from StatisticsUtils import CalculateAUC, ClassificationMetrics
import torch.nn as nn
import scipy.stats
import warnings

warnings.filterwarnings("ignore")
targetResolutionPerSubFigure = 1080
targetDPI = 200
from sklearn.metrics import f1_score

class SingleTaskClassificationAnswer():
    def __init__(self):
        self.Outputs = torch.Tensor()
        self.Labels = torch.Tensor()
        self.DataIndexes = []
        self.Accuracy = 0
        self.Recall = 0
        self.Precision = 0
        self.Specificity = 0
        self.Sensitivity = 0
        self.PPV = 0
        self.NPV = 0
        self.F1 = 0
        self.TrainLosses = None
        self.ValidationLosses = None


def DrawPlots(validationFPRs, validationTPRs, validationAUCs, \
              testFPRs, testTPRs, testAUCs, \
              ensembleFPR, ensembleTPR, ensembleAUC, \
              validationAnswers, saveFolderPath, numOfFold):
    gridSize = 2
    targetFigureSize = (
    targetResolutionPerSubFigure * gridSize / targetDPI, targetResolutionPerSubFigure * gridSize / targetDPI)
    plot.figure(figsize=targetFigureSize, dpi=targetDPI)
    plot.subplot(gridSize, gridSize, 1)
    for i in range(5):
        plot.title("Validation AUC by folds")
        plot.plot(validationFPRs[i], validationTPRs[i], alpha=0.7,
                  label=("Fold %d Val AUC = %0.3f" % (i, validationAUCs[i])))
        plot.legend(loc="lower right")
        plot.plot([0, 1], [0, 1], "r--")
        plot.xlim([0, 1])
        plot.ylim([0, 1.05])
        plot.ylabel("True Positive Rate")
        plot.xlabel("False Positive Rate")

    plot.subplot(gridSize, gridSize, 2)
    for i in range(5):
        plot.title("Test AUC by folds")
        plot.plot(testFPRs[i], testTPRs[i], alpha=0.7, label=("Fold %d Test AUC = %0.3f" % (i, testAUCs[i])))
        plot.legend(loc="lower right")
        plot.plot([0, 1], [0, 1], "r--")
        plot.xlim([0, 1])
        plot.ylim([0, 1.05])
        plot.ylabel("True Positive Rate")
        plot.xlabel("False Positive Rate")

    plot.subplot(gridSize, gridSize, 3)
    plot.title("Test AUC by ensemble")
    plot.plot(ensembleFPR, ensembleTPR, alpha=0.7, label="ROC curve (AUC= %0.3f)" % ensembleAUC)
    plot.legend(loc="lower right")
    plot.plot([0, 1], [0, 1], "r--")
    plot.xlim([0, 1])
    plot.ylim([0, 1.05])
    plot.ylabel("Sensitivity")
    plot.xlabel("1-Specificity")
    plot.savefig(os.path.join(saveFolderPath, "ROCCurvePlot.png"))

    if validationAnswers[0].TrainLosses is None:
        return
    hasLabelLoss = hasattr(validationAnswers[0], "TrainLabelLosses")
    gridSize = 4 if hasLabelLoss else 3
    targetFigureSize = (
    targetResolutionPerSubFigure * gridSize / targetDPI, targetResolutionPerSubFigure * gridSize / targetDPI)
    plot.figure(figsize=targetFigureSize, dpi=targetDPI)
    for i in range(5):
        plot.subplot(gridSize, gridSize, i + 1)
        plot.title("Fold %d Losses" % i)
        plot.plot(np.array(validationAnswers[i].TrainLosses), label="Train Loss")
        plot.plot(np.array(validationAnswers[i].ValidationLosses), label="Validation Loss")
        plot.legend(loc="upper right")
        plot.xlabel("Epoch")
        plot.ylabel("Loss")

        if hasLabelLoss:
            plot.subplot(gridSize, gridSize, i + 6)
            plot.title("Fold %d Label Losses" % i)
            plot.plot(np.array(validationAnswers[i].TrainLabelLosses), label="Train Label Loss")
            plot.plot(np.array(validationAnswers[i].ValidationLabelLosses), label="Validation Label Loss")
            plot.legend(loc="upper right")
            plot.xlabel("Epoch")
            plot.ylabel("Loss")
    plot.savefig(os.path.join(saveFolderPath, "LossesPlot.png"))


def SingleTaskEnsembleTest(testAnswers, saveFolderPath):
    foldPredict = np.array([testAnswer.Outputs[:, 1].numpy() for testAnswer in testAnswers])
    label = testAnswers[0].Labels.numpy()
    rawResults0 = np.mean(foldPredict, axis=0)  # 取平均
    rawResults = np.zeros_like(rawResults0)
    for i in range(len(label)):  # 360不对不对
        tttt = mean_confidence_interval(foldPredict[:, i], confidence=0.95)
        rawResults[i] = np.float64(tttt[0])
    predict = (rawResults > 0.5).astype(np.int16)
    P = (predict == 1).astype(np.int16)
    N = (predict == 0).astype(np.int16)
    TP = np.sum(P * label)
    FP = np.sum(P * (1 - label))
    TN = np.sum(N * (1 - label))
    FN = np.sum(N * label)
    accuracy, recall, precision, specificity, sensitivity, PPV, NPV, F1 = ClassificationMetrics(TP, FP, TN, FN)

    ensembleAUC, ensembleFPR, ensembleTPR = CalculateAUC(rawResults, label, multi=False, needThreshold=False)
    print("\nEnsemble Test Results:")
    print("AUC,%f\nAccuracy,%f\nRecall,%f\nPrecision,%f\nSpecificity,%f\nSensitivity,%f\nPPV,,%f\nNPV,,%f\nF1%f\n" % \
          (ensembleAUC, accuracy, recall, precision, specificity, sensitivity, PPV, NPV, F1))

    with open(os.path.join(saveFolderPath, "TestResults.csv"), mode="w", newline="") as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["Ensemble Test Results:"])
        csvWriter.writerow(
            ["AUC:", ensembleAUC, "Accuracy:", accuracy, "Recall:", recall, "Precision:", precision, "Specificity:",
             specificity, "Sensitivity:", sensitivity, "PPV:", PPV, "NPV:", NPV, "F1:", F1])
        csvWriter.writerow(["DataIndex", "Ensembled", "Fold1", "Fold2", "Fold3", "Fold4", "Fold5"])
        for i, dataIndex in enumerate(testAnswers[0].DataIndexes):
            csvWriter.writerow(
                [dataIndex, str(rawResults[i]), str(foldPredict[0][i]), str(foldPredict[1][i]), str(foldPredict[2][i]),
                 str(foldPredict[3][i]), str(foldPredict[4][i])])

    return ensembleAUC, ensembleFPR, ensembleTPR


def SingleTaskEnsembleTest_tes(testAnswers, saveFolderPath):
    foldPredict = np.array([testAnswer.Outputs[:, 1].numpy() for testAnswer in testAnswers])
    label = testAnswers[0].Labels.numpy()
    rawResults0 = np.mean(foldPredict, axis=0)  
    rawResults = np.zeros_like(rawResults0)
    for i in range(len(label)):
        tttt = mean_confidence_interval(foldPredict[:, i], confidence=0.95)
        rawResults[i] = np.float64(tttt[0])
    predict = (rawResults > 0.5).astype(np.int16)
    P = (predict == 1).astype(np.int16)
    N = (predict == 0).astype(np.int16)
    TP = np.sum(P * label)
    FP = np.sum(P * (1 - label))
    TN = np.sum(N * (1 - label))
    FN = np.sum(N * label)
    accuracy, recall, precision, specificity, sensitivity, PPV, NPV, F1 = ClassificationMetrics(TP, FP, TN, FN)
    ensembleAUC, ensembleFPR, ensembleTPR = CalculateAUC(rawResults, label, needThreshold=False, multi=False)
    print("\nEnsemble Test Results:")
    print("AUC,%f\nAccuracy,%f\nRecall,%f\nPrecision,%f\nSpecificity,%f\nSensitivity,%f\nPPV,,%f\nNPV,,%f\nF1%f\n" % \
          (ensembleAUC, accuracy, recall, precision, specificity, sensitivity, PPV, NPV, F1))

    with open(os.path.join(saveFolderPath, "TestResults_tes.csv"), mode="w", newline="") as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["Ensemble Test Results:"])
        csvWriter.writerow(
            ["AUC:", ensembleAUC, "Accuracy:", accuracy, "Recall:", recall, "Precision:", precision, "Specificity:",
             specificity, "Sensitivity:", sensitivity, "PPV:", PPV, "NPV:", NPV, "F1:", F1])
        csvWriter.writerow(["DataIndex", "Ensembled", "Fold1", "Fold2", "Fold3", "Fold4", "Fold5"])
        for i, dataIndex in enumerate(testAnswers[0].DataIndexes):
            csvWriter.writerow(
                [dataIndex, str(rawResults[i]), str(foldPredict[0][i]), str(foldPredict[1][i]), str(foldPredict[2][i]),
                 str(foldPredict[3][i]), str(foldPredict[4][i])])

    return ensembleAUC, ensembleFPR, ensembleTPR



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
    return m, m - h, m + h


def SingleTaskClassificationPrintAndPlot(validationAnswers, testAnswers, saveFolderPath):
    numOfFold = len(validationAnswers)
    validationAverages = [0] * 9  
    testAverages = [0] * 9  

    validationAUCs = []
    validationFPRs = []
    validationTPRs = []
    testAUCs = []
    testFPRs = []
    testTPRs = []

    valid_result = [[0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5]
    test_result = [[0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5]
    print(",,,Validation,,,,,,Test,,,,")
    print(
        "Fold,Accuracy,Recall,Precision,Specificity,AUC,F1,PPV,NPV,Sensitivity,,Accuracy,Recall,Precision,Specificity,AUC,F1,PPV,NPV,Sensitivity")
    for i in range(5):  # 5
        # Validation
        validationAUC, validationFPR, validationTPR = \
            CalculateAUC(validationAnswers[i].Outputs[:, 1].numpy(), validationAnswers[i].Labels.numpy(),
                         needThreshold=False, multi=False)
        validationAUCs.append(validationAUC)
        validationFPRs.append(validationFPR)
        validationTPRs.append(validationTPR)

        validationAverages[0] += validationAnswers[i].Accuracy
        validationAverages[1] += validationAnswers[i].Recall
        validationAverages[2] += validationAnswers[i].Precision
        validationAverages[3] += validationAnswers[i].Specificity
        validationAverages[4] += validationAUC
        validationAverages[5] += validationAnswers[i].F1
        validationAverages[6] += validationAnswers[i].PPV
        validationAverages[7] += validationAnswers[i].NPV
        validationAverages[8] += validationAnswers[i].Sensitivity

        valid_result[0][i] = validationAUC
        valid_result[1][i] = validationAnswers[i].Accuracy
        valid_result[2][i] = validationAnswers[i].Recall
        valid_result[3][i] = validationAnswers[i].Precision
        valid_result[4][i] = validationAnswers[i].Specificity
        valid_result[5][i] = validationAnswers[i].F1
        valid_result[6][i] = validationAnswers[i].PPV
        valid_result[7][i] = validationAnswers[i].NPV
        valid_result[8][i] = validationAnswers[i].Sensitivity

        print("%d," % i, end="")
        print("%f," % validationAnswers[i].Accuracy, end="")
        print("%f," % validationAnswers[i].Recall, end="")
        print("%f," % validationAnswers[i].Precision, end="")
        print("%f," % validationAnswers[i].Specificity, end="")
        print("%f,," % validationAUC, end="")
        print("%f,," % validationAnswers[i].F1, end="")
        print("%f,," % validationAnswers[i].PPV, end="")
        print("%f,," % validationAnswers[i].NPV, end="")
        print("%f,," % validationAnswers[i].Sensitivity, end="")

        # Test
        testAUC, testFPR, testTPR = \
            CalculateAUC(testAnswers[i].Outputs[:, 1].numpy(), testAnswers[i].Labels.numpy(), needThreshold=False,
                         multi=False)
        testAUCs.append(testAUC)
        testFPRs.append(testFPR)
        testTPRs.append(testTPR)

        testAverages[0] += testAnswers[i].Accuracy
        testAverages[1] += testAnswers[i].Recall
        testAverages[2] += testAnswers[i].Precision
        testAverages[3] += testAnswers[i].Specificity
        testAverages[4] += testAUC
        testAverages[5] += testAnswers[i].F1
        testAverages[6] += testAnswers[i].PPV
        testAverages[7] += testAnswers[i].NPV
        testAverages[8] += testAnswers[i].Sensitivity

        test_result[0][i] = testAUC
        test_result[1][i] = testAnswers[i].Accuracy
        test_result[2][i] = testAnswers[i].Recall
        test_result[3][i] = testAnswers[i].Precision
        test_result[4][i] = testAnswers[i].Specificity
        test_result[5][i] = testAnswers[i].F1
        test_result[6][i] = testAnswers[i].PPV
        test_result[7][i] = testAnswers[i].NPV
        test_result[8][i] = testAnswers[i].Sensitivity

        print("%f," % testAnswers[i].Accuracy, end="")
        print("%f," % testAnswers[i].Recall, end="")
        print("%f," % testAnswers[i].Precision, end="")
        print("%f," % testAnswers[i].Specificity, end="")
        print("%f," % testAUC, end="")
        print("%f,," % testAnswers[i].F1, end="")
        print("%f,," % testAnswers[i].PPV, end="")
        print("%f,," % testAnswers[i].NPV, end="")
        print("%f,," % testAnswers[i].Sensitivity, end="")
        print("\n")

    validationAverages = np.array(validationAverages) / numOfFold
    testAverages = np.array(testAverages) / numOfFold
    print("Average,", end="")
    for v in validationAverages:
        # mean_confidence_interval(logreg_scores, confidence=0.95)
        print("%f," % v, end="")
    print("！！！", end="")
    for v in testAverages:
        print("%f," % v, end="")
    print()

    print("95CI AUC, Average, Accuracy, Recall, Precision, Specificity, F1, PPV, NPV, Sensitivity\n", end="")
    for i in range(9):
        tmp = mean_confidence_interval(valid_result[i], confidence=0.95)
        print("%f " % np.float64(tmp[0]), end="")
        print("(%f, " % np.float64(tmp[1]), end="")
        print(" %f)! " % np.float64(tmp[2]), end="")

    print(" ")
    print("Test 95CI AUC, Average, Accuracy, Recall, Precision, Specificity, F1, PPV, NPV, Sensitivity\n", end="")
    for i in range(9):
        tmp = mean_confidence_interval(test_result[i], confidence=0.95)
        print("%f  " % np.float64(tmp[0]), end="")
        print("(%f, " % np.float64(tmp[1]), end="")
        print(" %f)! " % np.float64(tmp[2]), end="")
    print()

    ensembleAUC, ensembleFPR, ensembleTPR = SingleTaskEnsembleTest(testAnswers, saveFolderPath)

    DrawPlots(validationFPRs, validationTPRs, validationAUCs, \
              testFPRs, testTPRs, testAUCs, \
              ensembleFPR, ensembleTPR, ensembleAUC, \
              validationAnswers, saveFolderPath, numOfFold)



def ClassificationPrintAndPlot(validationAnswers, testAnswers, saveFolderPath):
    SingleTaskClassificationPrintAndPlot(validationAnswers, testAnswers, saveFolderPath)

def DrawPlots_tes(testFPRs, testTPRs, testAUCs, \
                  ensembleFPR, ensembleTPR, ensembleAUC, \
                  saveFolderPath, numOfFold):
    gridSize = 2
    targetFigureSize = (
    targetResolutionPerSubFigure * gridSize / targetDPI, targetResolutionPerSubFigure * gridSize / targetDPI)
    plot.figure(figsize=targetFigureSize, dpi=targetDPI)
    plot.subplot(gridSize, gridSize, 1)

    plot.subplot(gridSize, gridSize, 2)
    for i in range(5):  # 5
        plot.title("Test AUC by folds")
        plot.plot(testFPRs[i], testTPRs[i], alpha=0.7, label=("Fold %d Test AUC = %0.3f" % (i, testAUCs[i])))
        plot.legend(loc="lower right")
        plot.plot([0, 1], [0, 1], "r--")
        plot.xlim([0, 1])
        plot.ylim([0, 1.05])
        plot.ylabel("True Positive Rate")
        plot.xlabel("False Positive Rate")

    plot.subplot(gridSize, gridSize, 3)
    plot.title("Test AUC by ensemble")
    # plot.plot(ensembleFPR, ensembleTPR, alpha = 0.7, label = "Test AUC = %0.3f" % ensembleAUC)
    plot.plot(ensembleFPR, ensembleTPR, alpha=0.7, label="ROC curve (AUC= %0.3f)" % ensembleAUC)
    plot.legend(loc="lower right")
    plot.plot([0, 1], [0, 1], "r--")
    plot.xlim([0, 1])
    plot.ylim([0, 1.05])
    plot.ylabel("Sensitivity")
    plot.xlabel("1-Specificity")
    plot.savefig(os.path.join(saveFolderPath, "ROCCurvePlot_test.png"))

def SingleTaskClassificationPrintAndPlot_tes(testAnswers, saveFolderPath):
    numOfFold = len(testAnswers)
    testAverages = [0] * 9  
    testAUCs = []
    testFPRs = []
    testTPRs = []

    test_result = [[0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5]
    print(",,,Test,,,,")
    print(
        "Fold,Accuracy,Recall,Precision,Specificity,AUC,F1,PPV,NPV,Sensitivity,,Accuracy,Recall,Precision,Specificity,AUC,F1,PPV,NPV,Sensitivity")
    for i in range(5):  # numOfFold

        # Test
        testAUC, testFPR, testTPR = CalculateAUC(testAnswers[i].Outputs[:, 1].numpy(), testAnswers[i].Labels.numpy(),
                                                 needThreshold=False, multi=False)
        testAUCs.append(testAUC)
        testFPRs.append(testFPR)
        testTPRs.append(testTPR)

        testAverages[0] += testAnswers[i].Accuracy
        testAverages[1] += testAnswers[i].Recall
        testAverages[2] += testAnswers[i].Precision
        testAverages[3] += testAnswers[i].Specificity
        testAverages[4] += testAUC
        testAverages[5] += testAnswers[i].F1
        testAverages[6] += testAnswers[i].PPV
        testAverages[7] += testAnswers[i].NPV
        testAverages[8] += testAnswers[i].Sensitivity

        test_result[0][i] = testAUC
        test_result[1][i] = testAnswers[i].Accuracy
        test_result[2][i] = testAnswers[i].Recall
        test_result[3][i] = testAnswers[i].Precision
        test_result[4][i] = testAnswers[i].Specificity
        test_result[5][i] = testAnswers[i].F1
        test_result[6][i] = testAnswers[i].PPV
        test_result[7][i] = testAnswers[i].NPV
        test_result[8][i] = testAnswers[i].Sensitivity

        print("%f," % testAnswers[i].Accuracy, end="")
        print("%f," % testAnswers[i].Recall, end="")
        print("%f," % testAnswers[i].Precision, end="")
        print("%f," % testAnswers[i].Specificity, end="")
        print("%f," % testAUC, end="")
        print("%f,," % testAnswers[i].F1, end="")
        print("%f,," % testAnswers[i].PPV, end="")
        print("%f,," % testAnswers[i].NPV, end="")
        print("%f,," % testAnswers[i].Sensitivity, end="")
        print("\n")

    testAverages = np.array(testAverages) / numOfFold
    print("Average,", end="")
    print("Test 95CI AUC, Average, Accuracy, Recall, Precision, Specificity, F1, PPV, NPV, Sensitivity\n", end="")
    for i in range(9):
        tmp = mean_confidence_interval(test_result[i], confidence=0.95)
        print("%f  " % np.float64(tmp[0]), end="")
        print("(%f, " % np.float64(tmp[1]), end="")
        print(" %f)! " % np.float64(tmp[2]), end="")
    print()

    ensembleAUC, ensembleFPR, ensembleTPR = SingleTaskEnsembleTest_tes(testAnswers, saveFolderPath)

    DrawPlots_tes(testFPRs, testTPRs, testAUCs, \
                  ensembleFPR, ensembleTPR, ensembleAUC, \
                  saveFolderPath, numOfFold)

def ClassificationPrintAndPlot_tes(testAnswers, saveFolderPath):
    SingleTaskClassificationPrintAndPlot_tes(testAnswers, saveFolderPath)

