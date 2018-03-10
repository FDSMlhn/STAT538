import numpy as np



## KPCA

def RBF(x,y, sigma = 1):
    return np.exp(- np.linalg.norm(x-y)**2/(2*sigma))

def power_iter(X, iteration=500):
    v = np.random.randn(len(C))[:,np.newaxis]
    
    for i in range(iteration):
        temp_v = np.dot(C, v)
        v = temp_v / np.linalg.norm(temp_v)
        lam = np.dot(np.dot(v.T, C), v)
    return v, lam

def KPCA(X, kernel= RBF, n_components=2,iteration = 500, sigma=0):
    N = len(X)
#     GRAM = np.zeros((N,N))
#     for i in range(N):
#         for j in range(N):
#             GRAM[i,j] = kernel(X[i],X[j], **kwargs)
    
    
    X_norm = (X*X).sum(axis=1)
    norm_mat = X_norm[:,np.newaxis] + X_norm[np.newaxis,:]
    norm_mat -= 2 * np.dot(X,  X.T)
    GRAM = np.exp(-norm_mat/(2*sigma))
    print(GRAM)
    
    #norm_matrix = np.identity(N) - 1/len(X) * np.ones((len(X),len(X)))
    #C = np.dot(norm_matrix, np.dot(GRAM, norm_matrix))/(len(X)-1)
    C=GRAM
    C_copy = C.copy()
    print("Finished computing kernel matrix")
    # eigen_vectors = []
    # eigen_values = []
    first_n_components = []
    for i in range(n_components):
        v = np.random.randn(len(C))[:,np.newaxis]
        for i in range(iteration):
            temp_v = np.dot(C, v)
            v = temp_v / np.linalg.norm(temp_v)
            lam = np.dot(np.dot(v.T, C), v)
        # eigen_vectors.append(v)
        # eigen_values.append(lam)
        first_n_components.append(np.dot(C_copy, 1/np.sqrt(lam)*v))
        if i!=n_components-1:
            C -= lam* v * v.T
    return np.concatenate(first_n_components,axis=1)
#    return C_copy, eigen_vectors, eigen_values

# MLP

import torch 
import torch.nn as nn
import torch.nn.functional as F
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim,hu=1600):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hu)
        self.layer2 = nn.Linear(hu, hu)
        self.layer3 = nn.Linear(hu, output_dim)
    
    def forward(self,x):
        x1= F.relu(self.layer1(x), inplace = True)
        x2= F.relu(self.layer2(x1), inplace = True)
        x3= self.layer3(x2)
        return x1,x2,x3

    
    
class MLP2(nn.Module):
    def __init__(self, input_dim, output_dim,hu=1600):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hu)
        self.layer2 = nn.Linear(hu, hu)
        self.layer3 = nn.Linear(hu, output_dim)
    
    def forward(self,x):
        x1= F.sigmoid(self.layer1(x))
        x2= F.sigmoid(self.layer2(x1))
        x3= self.layer3(x2)
        return x1,x2,x3

        
class CNN(nn.Module):
    def __init__(self, input_dim, output_dim=10):
        super().__init__()
        self.layer1 = nn.Conv2d(input_dim, 100,kernel_size=5, padding=2, stride=1)
        self.pooling1= nn.MaxPool2d(kernel_size=3, stride=2,padding =1)
        
        self.layer2 = nn.Conv2d(100, 100,kernel_size=5, padding=2, stride=1)
        self.pooling2= nn.MaxPool2d(kernel_size=3, stride=2,padding =1)
        
        self.layer3 = nn.Conv2d(100, output_dim, kernel_size=7, padding=0)
    
    def forward(self,x):
        x1= self.pooling1(F.relu(self.layer1(x), inplace = True))
        x2= self.pooling2(F.relu(self.layer2(x1), inplace = True))
        x3= self.layer3(x2)
        return x1,x2,x3
    

    
    
    
    
    
    
    
    
    
    
    