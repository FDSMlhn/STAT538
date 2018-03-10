import torch
import torch.autograd as autograd
import numpy as np
import torch.utils.data
from torch import optim
from torch.autograd import Variable
#from joblib import Parallel, delayed
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F

def greedy_train(dbn, lr = [1e-3, 1e-4], epoch = [100, 100], batch_size = 50, input_data = None, weight_decay = [0,0], L1_penalty = [0,0], CD_k = 10, test_set = None, initialize_v = False):
    
    train_set = torch.utils.data.dataset.TensorDataset(input_data, torch.zeros(input_data.size()[0]))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    

    for i in range(dbn.n_layers):
        print("Training the %ith layer"%i)
        optimizer = optim.Adam(dbn.rbm_layers[i].parameters(), lr = lr[i], weight_decay = weight_decay[i])
        if initialize_v:
            v = Variable(input_data)
            for ith in range(i):
                p_v, v = dbn.rbm_layers[ith].v_to_h(v)
            dbn.rbm_layers[i].v_bias.data.zero_()
            dbn.rbm_layers[i].v_bias.data.add_(torch.log(v.mean(0)/(1-v.mean(0))).data)
        for _ in range(epoch[i]):
            for batch_idx, (data, target) in enumerate(train_loader):
                data = Variable(data)
                v, v_ = dbn(v_input = data, ith_layer = i, CD_k = CD_k)
                
                loss = dbn.rbm_layers[i].free_energy(v.detach()) - dbn.rbm_layers[i].free_energy(v_.detach()) + L1_penalty[0] * torch.sum(torch.abs(dbn.rbm_layers[i].W))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        if not type(test_set) == type(None):
            print("epoch %i: "%i, reconstruct_error(rbm, Variable(test_set)))
            


class RBM(nn.Module):
    def __init__(self,
                 n_visible=256,
                 n_hidden=64):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.Tensor(n_hidden,n_visible).uniform_(-1.0/(n_visible+n_hidden), 1.0/(n_visible+n_hidden)))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
    
    def sample_from_p(self,p):
        return torch.bernoulli(p)
    
    def v_to_h(self,v):
        
        # p_h = F.sigmoid(v.mm(self.W.t()) + self.h_bias.repeat(v.size()[0],1))
        p_h = F.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h
    
    def h_to_v(self,h):
        p_v = F.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v
        
    def forward(self,v, CD_k = 10):
        pre_h1,h1 = self.v_to_h(v)
        
        h_ = h1
        for _ in range(CD_k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)
        
        return v,v_
    
    def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = torch.clamp(F.linear(v,self.W,self.h_bias),-80,80)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

class DBN(nn.Module):
    def __init__(self,
                 n_visible=256,
                 n_hidden=[100,64],):
        super(DBN, self).__init__()
        
        self.n_layers = len(n_hidden)
        self.rbm_layers = []
        
        for i in range(self.n_layers):
            if i == 0:
                input_size = n_visible
            else:
                input_size = n_hidden[i-1]
            rbm = RBM(n_visible = input_size, n_hidden = n_hidden[i])
            
            self.rbm_layers.append(rbm)
        
        self.W_rec = [nn.Parameter(self.rbm_layers[i].W.data.clone()) for i in range(self.n_layers-1)]
        self.W_gen = [nn.Parameter(self.rbm_layers[i].W.data) for i in range(self.n_layers-1)]
        self.bias_rec = [nn.Parameter(self.rbm_layers[i].h_bias.data.clone()) for i in range(self.n_layers-1)]
        self.bias_gen = [nn.Parameter(self.rbm_layers[i].v_bias.data) for i in range(self.n_layers-1)]
        self.W_mem = nn.Parameter(self.rbm_layers[-1].W.data)
        self.v_bias_mem = nn.Parameter(self.rbm_layers[-1].v_bias.data)
        self.h_bias_mem = nn.Parameter(self.rbm_layers[-1].h_bias.data)
        
        for i in range(self.n_layers-1):
            self.register_parameter('W_rec%i'%i, self.W_rec[i])
            self.register_parameter('W_gen%i'%i, self.W_gen[i])
            self.register_parameter('bias_rec%i'%i, self.bias_rec[i])
            self.register_parameter('bias_gen%i'%i, self.bias_gen[i])
        
        
    def forward(self, v_input, ith_layer, CD_k = 10): #for greedy training
        v = v_input
        
        for ith in range(ith_layer):
            p_v, v = self.rbm_layers[ith].v_to_h(v)
            
        v, v_ = self.rbm_layers[ith_layer](v, CD_k = CD_k)

        return v, v_