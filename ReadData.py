import csv
import random
import numpy as np
random.seed(2223)
from openpyxl import load_workbook as lw
import cv2
class Folds:
    def __init__(self, benignPatients, malignantPatients, numOfFold, dataPath, patients):
        self.BenignPatients = benignPatients
        self.MalignantPatients = malignantPatients
        self.NumOfFold = numOfFold
        self.DataPath = dataPath
        self.Ratio = 0.11#0.11
        self.patients = patients

    def LoadSetData(self, benignStart, benignEnd, malignantStart, malignantEnd, dataPercentage = 1.0):
        setDataIndexList = []
        setLabelList = []
        def LoadBenignOrMalignant(start, end, ps):
            numOfPatients = int((end - start) * dataPercentage)
            for i in range(start, start + numOfPatients):
                setDataIndexList.append(ps[i][0])
                setLabelList.append(ps[i][1:])
        LoadBenignOrMalignant(int(benignStart), int(benignEnd), self.BenignPatients)
        LoadBenignOrMalignant(int(malignantStart), int(malignantEnd), self.MalignantPatients)
        return np.array(setDataIndexList), np.array(setLabelList)

    def LoadBenignOrMalignant(self, start, end):
        setDataIndexList = []
        setLabelList = []
        ps = self.patients
        numOfPatients = int((end - start) * 1.0)
        for i in range(start, start + numOfPatients):
            setDataIndexList.append(ps[i][0])
            setLabelList.append(ps[i][1:])
        return np.array(setDataIndexList), np.array(setLabelList)
    def NextFold(self, trainDataPercentage = 1.0):

        fold = {}
        whole = {}
        random.shuffle(self.BenignPatients)  # 随机打乱顺序
        random.shuffle(self.MalignantPatients)  # 随机打乱顺序
        random.shuffle(self.patients)
        whole["DataPath"] = self.DataPath
        whole["ValidationSetDataIndex"], \
        whole["ValidationSetLabel"] = \
            self.LoadSetData(0, round(len(self.BenignPatients) * self.Ratio), \
                             0, round(len(self.MalignantPatients) * self.Ratio))
        fold["DataPath"] = self.DataPath
        fold["TrainSetDataIndex"], \
        fold["TrainSetLabel"] = \
            self.LoadSetData(round(len(self.BenignPatients) * self.Ratio), len(self.BenignPatients), \
                             round(len(self.MalignantPatients) * self.Ratio), len(self.MalignantPatients), \
                             dataPercentage=trainDataPercentage)

        self.BenignPatients = self.BenignPatients[
                              round(len(self.BenignPatients) * self.Ratio): len(self.BenignPatients)] + \
                              self.BenignPatients[0: round(len(self.BenignPatients) * self.Ratio)]
        self.MalignantPatients = self.MalignantPatients[
                                 round(len(self.MalignantPatients) * self.Ratio): len(self.MalignantPatients)] + \
                                 self.MalignantPatients[0: round(len(self.MalignantPatients) * self.Ratio)]
        return fold,whole


    def GetWholeAsVal(self):
        whole = {}
        random.shuffle(self.patients)
        whole["DataPath"] = self.DataPath
        whole["ValidationSetDataIndex"], \
        whole["ValidationSetLabel"] = \
            self.LoadBenignOrMalignant(0, len(self.patients))
        return whole
    def GetWholeAsTest(self):
        whole = {}
        whole["DataPath"] = self.DataPath
        whole["TestSetDataIndex"], \
        whole["TestSetLabel"] = \
            self.LoadBenignOrMalignant(0, len(self.patients))
        return whole


def ReadFolds(paths):
    dataPath, infoPath = paths
    patients = []
    sheet = lw(infoPath).worksheets[0]
    pass_title = False
    for row in sheet.values:
        tmp_1 = []
        if not pass_title:
            pass_title = True
            continue
        dataIndex = str(row[0]) 
        label = int(row[1])  
        for j in range(2, 50):
            tmp_1_0 = float(row[j])
            tmp_1.append(tmp_1_0)

        patients.append((dataIndex,label,tmp_1[0], tmp_1[1], tmp_1[2],tmp_1[4], tmp_1[14], tmp_1[24], tmp_1[47],))

    benignPatients = []
    malignantPatients = []
    for patient in patients:
        if patient[1] == 0:
            benignPatients.append(patient)
        else:
            malignantPatients.append(patient)
    folds = Folds(benignPatients, malignantPatients, 5, dataPath, patients)
    return folds
