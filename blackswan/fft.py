import torch

import numpy as np
from torch import nn
from matplotlib import pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"
if  not torch.cuda.is_available():
    print("not Using the GPU!")


def cal_functional(data_name,dim1,dim2):
    """
    data_name = str
    """
    if data_name.startswith("./"):
        data = torch.load(data_name)
    else:
        data = torch.load("./dinofeat/{}.pt".format(data_name)) # f"{data}".pt"  #data (1,256,384)

    data = data.permute(2,1,0) 
    data = data.reshape((384,16,16))
    transformed_features = np.fft.fft2(data) # (384,16,16)
    transformed_features = transformed_features[:,:dim1,:dim2].transpose(1,2,0) # (dim1,dim2,384)
   
   
    return transformed_features

def cal_func_map(data1,data2, is_real,dim1,dim2):
    """
    data1: (dim1,dim2,384)
    data2: (dim1,dim2,384)
    is_real: True or False
    """
    data_1 = cal_functional(str(data1).zfill(5),dim1,dim2)
    data_2 = cal_functional(str(data2).zfill(5),dim1,dim2)
    
   
    
    T = nn.Parameter(torch.randn(dim1*dim2,dim1*dim2))

    loss_fn = nn.HuberLoss(delta = 1.0)
    optimizer = torch.optim.Adam([T],lr = 0.001)
    data_1 = torch.from_numpy(data_1.real).float() if is_real else torch.from_numpy(data_1.imag).float() # (dim1,dim2,384)
    data_2 = torch.from_numpy(data_2.real).float() if is_real else torch.from_numpy(data_2.imag).float()
    data_1 = data_1.reshape((dim1*dim2,-1)) #(dim1*dim2,384)
    data_2 = data_2.reshape((dim1*dim2,-1)) #(dim1*dim2,384)
    epoch = 10000
    losses = []
    
    for i in range(epoch):
        optimizer.zero_grad()
        # data_0 (30,384) T(30,30)
        data_1_mul_T = torch.matmul(T,data_1) 
        loss = loss_fn(data_1_mul_T,data_2)
        loss.backward()
        optimizer.step()
        if i%1000 == 0:
            print("epoch:{},loss:{}".format(i,loss.item()))
            losses.append(loss.item())
    torch.save(T,"./dinofeat/functional_map_{}_{}_{}.pt".format(int(data1),int(data2),is_real))
    plt.plot(losses)
    plt.savefig(f"loss.png")
   
 
    
if __name__ == "__main__":
    f0 = cal_functional("00000",5,5)

    
    cal_func_map(0,1, True,5,5)
    cal_func_map(0,1, False,5,5)
    T = torch.complex(torch.load("./dinofeat/functional_map_0_49_True.pt").double(),torch.load("./dinofeat/functional_map_0_49_False.pt").double())
    '''
    data_1 = cal_functional(str(0).zfill(5),5,5)
    data_49 = cal_functional(str(49).zfill(5),5,5)
    print(torch.matmul(T,torch.tensor(data_1).reshape((25,-1))) - torch.tensor(data_49).reshape((25,-1)))
    print(torch.tensor(data_49))
    '''
    
