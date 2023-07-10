import torch

import numpy as np


def cal_functional(data_name):
    """
    data_name = 00000, 00001, ...
    """
    data = torch.load("./dinofeat/{}.pt".format(data_name)) # f"{data}".pt"
    data = data.permute(2,0,1)
    transformed_features = np.fft.fft2(data)
   
    transformed_features = transformed_features[:,0,:30]
   
    return transformed_features

def cal_func_map(data1,data2, is_real):
    """
    data1: (384,30)
    data2: (384,30)
    is_real: True or False
    """
    data_1 = cal_functional(str(data1).zfill(5))
    data_2 = cal_functional(str(data2).zfill(5))
    from torch import nn
    T = nn.Parameter(torch.randn(30,30))
    loss_fn = nn.MSELoss() # mean square error
    optimizer = torch.optim.Adam([T],lr = 0.001)
    data_1 = torch.from_numpy(data_1.real).float() if is_real else torch.from_numpy(data_1.imag).float()
    data_2 = torch.from_numpy(data_2.real).float() if is_real else torch.from_numpy(data_2.imag).float()
    
    epoch = 10000
    
    for i in range(epoch):
        optimizer.zero_grad()
        # data_0 (384,30) T(30,30) data_0_mul_T(384,30)
        data_1_mul_T = torch.matmul(data_1,T) 
        loss = loss_fn(data_1_mul_T,data_2)
        loss.backward()
        optimizer.step()
        if i%1000 == 0:
            print("epoch:{},loss:{}".format(i,loss.item()))
    torch.save(T,"./dinofeat/functional_map_{}_{}.pt".format(int(data1),int(data2)))
cal_func_map(0,7,True)

