import os
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

UltrasoundDataTransform_b = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([256, 256]),
    transforms.RandomRotation(15),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.ToTensor()#,
])

UltrasoundDataTransform2_b = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([224, 224]),
    transforms.ToTensor()#,

])

class TensorDatasetWithTransform(Dataset):
    def __init__(self, tensors, dataFolderPath, dataIndexes, transform):
        self.Tensors = tensors
        self.Transform = transform
        self.DataFolderPath = dataFolderPath
        self.DataIndexes = dataIndexes

    def __getitem__(self, index):
        dataIndex = self.DataIndexes[index]
        image_new = self.Tensors[0][index]
        label = self.Tensors[0][index]
        return image_new, label, dataIndex
    def pre_process(self, image):
        res = cv2.bilateralFilter(image, 5, 31, 31)
        hist, bins = np.histogram(res.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        res = cdf[res]
        return res
    def __len__(self):
        return self.Tensors[0].size(0)
