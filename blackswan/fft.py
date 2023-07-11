import torch

import numpy as np



def cal_functional(data_name):
    """
    data_name = str
    """
    if data_name.startswith("./"):
        data = torch.load(data_name)
    else:
        data = torch.load("./dinofeat/{}.pt".format(data_name)) # f"{data}".pt"  #data (1,256,384)

    data = data.permute(1,2,0) 
    transformed_features = np.fft.fft2(data) # (256,384,1)
    transformed_features = transformed_features[:,:30,0].transpose(1,0) # (384,256)
   
   
    return transformed_features

def cal_func_map(data1,data2, is_real):
    """
    data1: (1,256,384)
    data2: (1,256,384)
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
        data_1_mul_T = torch.matmul(T,data_1) 
        loss = loss_fn(data_1_mul_T,data_2)
        loss.backward()
        optimizer.step()
        if i%1000 == 0:
            print("epoch:{},loss:{}".format(i,loss.item()))
    torch.save(T,"./dinofeat/functional_map_{}_{}_{}.pt".format(int(data1),int(data2),is_real))
 
    
if __name__ == "__main__":
    cal_func_map(0,49, True)
    cal_func_map(0,49, False)
    T = torch.load("./dinofeat/functional_map_0_49_True.pt")
    from matplotlib import pyplot as plt
    plt.imshow(T.detach().numpy())
    plt.savefig("./dinofeat/functional_map_0_49_True.png")

