import os
import time
import torch
import shutil
import numpy as np
import torch.nn as nn
import torch.utils.data as TorchData
import ReadData as ReadData
import PrintAndPlot as PrintAndPlot
from CustomedDataset import TensorDatasetWithTransform, UltrasoundDataTransform_b, UltrasoundDataTransform2_b


class MachineLearningModel:
    def __init__(self, earlyStoppingPatience, learnRate, batchSize):
        self.EarlyStoppingPatience = earlyStoppingPatience
        self.LearnRate = learnRate
        self.BatchSize = batchSize
        self.Epsilon = 1e-6
        self.BestStateDict = None
        self.SetupSeed(2223)

    def SetupSeed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def LoadStateDictionary(self, stateDictionary):
        self.Net.load_state_dict(stateDictionary, strict=True)

    def LoadSet(self, set, fold, transform):
        setDataIndex = fold[set + "SetDataIndex"]
        setLabel = torch.from_numpy(fold[set + "SetLabel"]).long()
        setDataset = TensorDatasetWithTransform([setLabel],
                                                fold["DataPath"],
                                                setDataIndex,
                                                transform=transform)
        setLoader = TorchData.DataLoader(setDataset, batch_size=self.BatchSize, num_workers=0,
                                         shuffle=False if set == "Test" else True)  
        return setDataIndex, setLabel, setLoader

    def LoadData(self, fold1, fold2):
        self.TrainDataIndex, self.TrainLabel, self.TrainLoader = self.LoadSet("Train", fold1,UltrasoundDataTransform_b)
        self.ValidationDataIndex, self.ValidationLabel, self.ValidationLoader = self.LoadSet("Validation", fold2, UltrasoundDataTransform2_b)


def EvaluateMachineLearningModel(modelClass, \
                                 saveFolderPath, trainPaths, testPaths, \
                                 earlyStoppingPatience=30, learnRate=0.00001, batchSize=16, \
                                 name=None, numclass=2):  # 64earlyStoppingPatience = 20
    folds = ReadData.ReadFolds(trainPaths)
    testFold = ReadData.ReadFolds(testPaths)
    saveFolderPath = os.path.join(saveFolderPath, modelClass.__name__ + ("" if name is None else name) + time.strftime("%Y-%m-%d %H.%M.%S", time.localtime()))
    os.mkdir(saveFolderPath)
    validationAnswers = []
    testAnswers = []
    for i in range(5):  
        print("-------------------------------------------------------------")
        print("fold:", i)
        print("-------------------------------------------------------------")
        model = modelClass(earlyStoppingPatience=earlyStoppingPatience, learnRate=learnRate, batchSize=batchSize,numclass=numclass)
        model.Net = model.Net.cuda()
        model.LossFunction = model.LossFunction.cuda()
        fold1_train, fold2_val = folds.NextFold()
        model.LoadData(fold1_train, fold2_val) 
        validationAnswer, BestStateDict1 = model.Train()
        validationAnswers.append(validationAnswer)
        model.TestDataIndex, model.TestLabel, model.TestLoader = model.LoadSet("Test", testFold.GetWholeAsTest(),  UltrasoundDataTransform2_b)  
        testAnswer, _ = model.Evaluate(model.TestLoader, BestStateDict1) 
        testAnswers.append(testAnswer)
        torch.save(model.BestStateDict, os.path.join(saveFolderPath, "Fold%dWeights.pkl" % i))

    PrintAndPlot.ClassificationPrintAndPlot(validationAnswers, testAnswers, saveFolderPath)
    
