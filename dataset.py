import torch
from torch.utils import data
import torch.nn.functional as F

class MyDataset(data.Dataset):
    def __init__(self, data, label, context_size = 1):
        # data and label nparrays directly loaded from files
        N = len(data)
        X = []
        Y = []
        locator = []
        ctr = 0
        for i in range(N):
            xi = torch.from_numpy(data[i]).float()
            yi = torch.from_numpy(label[i]).long()
            X.append(xi)
            Y.append(yi)
            li = xi.shape[0]
            for j in range(li):
                locator.append((i, j)) #ith piece, jth time frame
                ctr += 1
        
        self.X = X
        self.Y = Y
        self.locator = locator
        self.context_size = context_size

    def __len__(self):
        return 100 #len(self.locator)

    def __getitem__(self,index):
        i, j = self.locator[index]
        left = j - self.context_size
        right = j + self.context_size # both sides inclusive
        X = self.X[i][j:j+1]
        if left < 0:
            XL = self.X[i][:j]
            XL = F.pad(input=XL, pad=(0, 0, self.context_size-XL.shape[0], 0), mode='constant', value=0)
        else:
            XL = self.X[i][left:j]
        
        if right >= self.X[i].shape[0]:
            
            XR = self.X[i][j+1:]
            XR = F.pad(input=XR, pad=(0, 0, 0, self.context_size-XR.shape[0]), mode='constant', value=0)
        else:
            XR = self.X[i][j+1:right+1]
        
        X = torch.cat((XL,X,XR), 0).reshape(-1)
        Y = self.Y[i][j]
        
        return X,Y