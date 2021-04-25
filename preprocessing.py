import numpy as np 
import csv
import torch 
import torch.nn as nn
from typing import List 
from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms  

# Preprocessing Data
class Preprocess(Dataset):
    @staticmethod
    def setlabels():
        labs = list(range(25))
        labs.pop(9)
        return labs 
    
    @staticmethod 
    def labels_from_csv(path):
        labs = Preprocess.setlabels()
        labels, samples = [], []
        with open(path) as f:
            _ = next(f)
            for l in csv.reader(f):
                label = int(l[0])
                labels.append(labs.index(label))
                samples.append(list(map(int, l[1:])))
        return labels, samples 
    
    def __init__(self, path: str = './data/sign_mnist_train.csv', mean: List[float] = [0.485], std: List[float] = [0.229]):
        labels, samples = Preprocess.labels_from_csv(path)
        self._samples = np.array(samples, dtype=np.uint8).reshape(-1,28,28,1)
        self._labels = np.array(labels, dtype = np.uint8).reshape((-1,1))

        self._mean = mean
        self._std = std 
        
    def __len__(self):
        return len(self._labels)
        
    def __getitem__(self, idx):
        '''
        we normalize the pictures here so it ensures that each input parameter (pixel, in this case) 
        has a similar data distribution. This makes convergence faster while training the network.
        '''
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(28, scale = (0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean = self._mean, std=self._std)
        ])

        return {'image': transform(self._samples[idx]).float(),'label': torch.from_numpy(self._labels[idx]).float()}

def train_test_loaders(batch_size=32):
    trainset = Preprocess('./data/sign_mnist_train.csv')
    train_load = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = Preprocess('./data/sign_mnist_test.csv')
    test_load = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return train_load, test_load   

if __name__ == '__main__':
    train_load, test_load = train_test_loaders(2)
    print(next(iter(train_load)))
