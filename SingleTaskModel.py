import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from termcolor import colored
from PrintAndPlot import SingleTaskClassificationAnswer
import torch.nn.functional as F
from model import DiffusionProcess, ConditionalDiffusion_CycleGAN_tequan_Mode
from StatisticsUtils import ClassificationMetrics, BinaryClassificationMetric
from MachineLearningModel import MachineLearningModel, EvaluateMachineLearningModel
import time
import warnings
from StatisticsUtils import CalculateAUC

# 固定随机数种子
seed = 42  # 选择一个你喜欢的数字作为种子
torch.manual_seed(seed)
T = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffusion = DiffusionProcess(T=T, device=device)
warnings.filterwarnings("ignore")


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1):  # beta=1.0
        super().__init__()
        self.alpha = alpha  # 分类损失权重
        self.beta = beta  # 对抗损失权重
        self.ce_loss = nn.CrossEntropyLoss()#nn.CrossEntropyLoss(weight=weight)

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()


    def forward(self, outputs, labels, clinical,ni_input_labels):
        # 分类损失
        loss_real = self.ce_loss(outputs['output_real'], labels)
        loss_fake = self.ce_loss(outputs['output_fake'], labels)
        cls_loss = (loss_real + 4*loss_fake) / 2  # 4*
        # 对抗损失
        adv_loss = self.bce_loss(outputs['validity_fake'], torch.zeros_like(outputs['validity_fake']))
        adv_loss2 = self.bce_loss(outputs['validity_real'], torch.ones_like(outputs['validity_real']))
        # 特征对齐损失
        align_loss = 4 * self.mse_loss(outputs['hallucinated'], clinical)#4*

        #A-B的损失
        loss_ni_real = self.ce_loss(outputs['output_ni_real'], ni_input_labels)
        loss_ni_fake = self.ce_loss(outputs['output_ni_fake'], ni_input_labels)
        cls_loss_ni=(loss_ni_real + 1*loss_ni_fake) / 28

        total_loss =6* cls_loss + align_loss + 4 * adv_loss + 1* adv_loss2+1 *cls_loss_ni#6 * cls_loss#7*cls_loss
        return total_loss  # .item()


class MultiTaskLoss_test(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):  # beta=1.0
        super().__init__()
        self.alpha = alpha  # 分类损失权重
        self.beta = beta  # 对抗损失权重
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()


    def forward(self, outputs, labels):
        cls_loss = self.ce_loss(outputs['output_fake'], labels)
        # 对抗损失
        adv_loss = self.bce_loss(outputs['validity_fake'], torch.zeros_like(outputs['validity_fake']))
        total_loss = 1 * cls_loss + self.beta * adv_loss
        return total_loss


class SingleTaskModel(MachineLearningModel):
    def __init__(self, earlyStoppingPatience, learnRate, batchSize, numclass):
        super().__init__(earlyStoppingPatience, learnRate, batchSize)
        #我们提出的方法
        self.Net = ConditionalDiffusion_CycleGAN_tequan_Model(input_dim=7, output_dim=2)
        self.LossFunction = MultiTaskLoss_test()
        self.train_loss_criterion = MultiTaskLoss()

    def Train(self):
        epoch = 0
        patience = self.EarlyStoppingPatience
        optimizer = torch.optim.Adam(self.Net.parameters(), lr=self.LearnRate)  # ,weight_decay=0.01,, weight_decay=l2_decay
        numOfInstance = len(self.TrainLabel)
        minLoss = float(0x7FFFFFFF)
        maxAUC =0
        bestValidationAnswer = None
        trainLosses = []
        validationLosses = []

        while patience > 0:#for i in range(500):
            self.Net.train()
            epoch += 1
            runningLoss = 0.0
            for batchImage, batchLabel, _ in self.TrainLoader:#64,3,224,224

                batchLabel1 = batchLabel[:, 0].long().cuda()#夜间高血压 0或1
                batchImage = batchLabel[:, 1:].float().cuda()
                target_onehot = F.one_hot(batchLabel1, num_classes=2).float().cuda()
                batch_size = batchLabel.size(0)
                device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
                # 随机采样时间步
                t = torch.randint(0, diffusion.T, (batch_size,)).cuda()
                # 前向扩散过程
                noise_input = torch.randn_like(batchImage, dtype=torch.float32).cuda()
                noisy_input_target, true_noise = diffusion.forward_diffusion(batchImage, t, noise_input)
                noise = torch.randn_like(target_onehot, dtype=torch.float32).cuda()
                noisy_target, true_noise = diffusion.forward_diffusion(target_onehot, t, noise)
                # 预测噪声
                outputClass = self.Net.forward(noisy_target, batchImage, t, train_falg=0)
                loss = self.train_loss_criterion(outputClass, batchLabel1, noisy_target,batchImage)
                optimizer.zero_grad()
                loss = loss.mean()
                runningLoss += loss.item()
                loss.backward()
                optimizer.step()

            self.Net.eval()
            trainLoss = (runningLoss * self.BatchSize) / numOfInstance
            validationAnswer, validationLoss = self.Evaluate(self.ValidationLoader, None)

            trainLosses.append(trainLoss)
            validationLosses.append(validationLoss)
            print("Epoch %d:  (Patience left: %d )\ntrainLoss -> %.3f, valLoss -> %.3f" % (epoch, patience, trainLoss, validationLoss))
            print("Accuracy -> %f" % validationAnswer.Accuracy, end = ", ")
            print("Recall -> %f" % validationAnswer.Recall, end = ", ")
            print("Precision -> %f" % validationAnswer.Precision, end = ", ")
            print("Sensitivity -> %f" % validationAnswer.Sensitivity, end = ", ")
            print("Specificity -> %f" % validationAnswer.Specificity)
            print("f1_score -> %f" % validationAnswer.F1)
            print("PPV -> %f" % validationAnswer.PPV)
            print("NPV -> %f" % validationAnswer.NPV)
            validationAUC, validationFPR, validationTPR = CalculateAUC(validationAnswer.Outputs[:, 1].numpy(), validationAnswer.Labels.numpy(), needThreshold=False,multi=False)
            if minLoss > validationLoss:
                patience = self.EarlyStoppingPatience
                minLoss = validationLoss
                bestValidationAnswer = validationAnswer
                self.BestStateDict = copy.deepcopy(self.Net.state_dict())
                print(colored("Better!!!!!!!!!!!!!!!!!!!!!!!!!!!", "green"))
            else:
                patience -= 1
                print(colored("Worse!!!!!!!!!!!!!!!!!!!!!!!!!!!", "red"))

        bestValidationAnswer.TrainLosses = trainLosses
        bestValidationAnswer.ValidationLosses = validationLosses
        return bestValidationAnswer,self.BestStateDict

    def Evaluate(self, dataLoader, stateDictionary):
        self.Net.eval()
        answer = SingleTaskClassificationAnswer()
        if stateDictionary is not None:
            self.LoadStateDictionary(stateDictionary)
        with torch.no_grad():
            numOfInstance = len(dataLoader.dataset)
            runningLoss = 0.0
            TP = [0]
            FP = [0]
            TN = [0]
            FN = [0]
            for batchImage, batchLabel, batchDataIndex in dataLoader:
                batchLabel1 = batchLabel[:, 0].long().cuda()  # 良性or恶性
                batchImage= batchLabel[:, 1:].float().cuda()  # 良性or恶性
                batch_size, _ = batchLabel.shape
                target_onehot = F.one_hot(batchLabel1, num_classes=2).float().cuda()
                # 随机采样时间步
                t = torch.randint(0, diffusion.T, (batch_size,)).cuda()
                noise = torch.randn_like(target_onehot, dtype=torch.float32).cuda()
                noisy_target, true_noise = diffusion.forward_diffusion(target_onehot, t, noise)
                # 预测噪声
                outputClass = self.Net.forward(noisy_target, batchImage, t,train_falg=1)
                loss = self.LossFunction(outputClass, batchLabel1)
                outputClass = outputClass['output_fake']
                loss = loss.mean()  # + loss_ori.mean()
                runningLoss += loss.item()
                BinaryClassificationMetric(outputClass, batchLabel1, TP, FP, TN, FN, 0)
                answer.Outputs = torch.cat((answer.Outputs, outputClass.softmax(dim=1).cpu()), dim=0)
                answer.Labels = torch.cat((answer.Labels, batchLabel1.float().cpu()), dim=0)
                answer.DataIndexes += batchDataIndex

            print("batchLabel1:{}".format(batchLabel1))
            answer.Accuracy, answer.Recall, answer.Precision, answer.Specificity, answer.Sensitivity, answer.PPV, answer.NPV, answer.F1 = ClassificationMetrics(
                TP[0], FP[0], TN[0], FN[0], self.Epsilon)

            loss = (runningLoss * self.BatchSize) / numOfInstance
            if stateDictionary is not None:                #if ss == "Test":
                print("batchLabel1:{}".format(batchLabel1))
                print("Accuracy -> %f" % answer.Accuracy, end=", ")
                print("Recall -> %f" % answer.Recall, end=", ")
                print("Precision -> %f" % answer.Precision, end=", ")
                print("Sensitivity -> %f" % answer.Sensitivity, end=", ")
                print("Specificity -> %f" % answer.Specificity)
                print("PPV -> %f" % answer.PPV)
                print("NPV -> %f" % answer.NPV)
                print("f1_score -> %f" % answer.F1)
                testAUC, testFPR, testTPR = CalculateAUC(
                    answer.Outputs[:, 1].numpy(), answer.Labels.numpy(), needThreshold=False,
                    multi=False)
                print("testAUC -> %f" % testAUC)

            return answer, loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--TrainFolderPath", help = "define the train data folder path", type = str)
    parser.add_argument("--TestFolderPath", help = "define the test data folder path", type = str)
    parser.add_argument("--TrainInfoPath", help = "define the train info path", type = str)
    parser.add_argument("--TestInfoPath", help = "define the test info path", type = str)
    parser.add_argument("--SaveFolderPath", help = "define the save folder path", type = str)
    parser.add_argument("--Name", help = "define the name", type = str)
    args = parser.parse_args()
    args.TrainFolderPath = " "
    args.TrainInfoPath = "NH-train.xlsx"
    args.TestFolderPath = " "
    args.TestInfoPath = "NH-test.xlsx"  #
    args.SaveFolderPath = r"./save_model"
    print("夜间高血压2分类")
    args.Name = "NH_model" + time.strftime("%Y-%m-%d %H.%M.%S",time.localtime())
    EvaluateMachineLearningModel(SingleTaskModel, \
                                 args.SaveFolderPath, (args.TrainFolderPath, args.TrainInfoPath),
                                 (args.TestFolderPath, args.TestInfoPath), earlyStoppingPatience=30, batchSize=16, \
                                 name=args.Name,numclass=2)
