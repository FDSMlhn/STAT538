import numpy as np



## KPCA

def RBF(x,y, sigma = 1):
    return np.exp(- np.linalg.norm(x-y)**2/(2*sigma))

def power_iter(X, iteration=500):
    C = np.dot(np.dot(X.T, np.identity(len(X)) - 1/len(X) * np.ones((len(X),len(X)))), X)/(len(X)-1)

    v = np.random.randn(len(C))[:,np.newaxis]
    
    for i in range(iteration):
        temp_v = np.dot(C, v)
        v = temp_v / np.linalg.norm(temp_v)
        lam = np.dot(np.dot(v.T, C), v)
    return v, lam

def KPCA(X, kernel= RBF, n_components=2,iteration = 500, **kwargs):
    N = len(X)
    GRAM = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            GRAM[i,j] = kernel(X[i],X[j], **kwargs)
    
    norm_matrix = np.identity(N) - 1/len(X) * np.ones((len(X),len(X)))
    C = np.dot(norm_matrix, np.dot(GRAM, norm_matrix))/(len(X)-1)
    C_copy = C.copy()
    
    eigen_vectors = []
    eigen_values = []
    for i in range(n_components):
        v = np.random.randn(len(C))[:,np.newaxis]
        for i in range(iteration):
            temp_v = np.dot(C, v)
            v = temp_v / np.linalg.norm(temp_v)
            lam = np.dot(np.dot(v.T, C), v)
        eigen_vectors.append(v)
        eigen_values.append(lam)
        if i!=n_components-1:
            C -= eigen_value* v * v.T
    return C_copy, eigen_vectors, eigen_values

# MLP

import torch 
import torch.nn as nn
import torch.nn.functional as F
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 1600)
        self.layer2 = nn.Linear(1600, 1600)
        self.layer3 = nn.Linear(1600, output_dim)
    
    def forward(self,x):
        x1= F.relu(self.layer1(x), inplace = True)
        x2= F.relu(self.layer2(x1), inplace = True)
        x3= self.layer3(x2)
        return x1,x2,x3
        


    
    
    
    
    
    
    
    
    
    
    