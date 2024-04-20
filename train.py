import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from matplotlib import pyplot as plt

#load data
class sickSkinDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform= None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

# returns the length of how much data we have in the dataset    
    def __len__(self):
        return len(self.annotations)

# will return a specific example corresponding to the target image
    def __getItem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.))


# prepare data

#train

#train classifier

#test performance
