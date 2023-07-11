import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from pca import pcaa, pca_nocorespondence
pca = PCA(n_components=3)

head = torch.zeros((16,16,384))
head[4][8] = 1
head[3][8] = 1
head[4][9] = 1
head[3][9] = 1
head[2][9] = 1
head[2][8] = 1
for i in range(5,9):
    head[i][8] = 1
for i in range(8,13):
    for j in range(2,10):
        head[i][j] = 1
head = head.reshape((1,256,384))
torch.save(head, "./gendata/head_plot.pt")
from pca import pca_nocorespondence
pca_nocorespondence("./gendata/head_plot.pt")

from fft import cal_functional
torch.save(head, "./dinofeat/head_plot.pt")
feat_0 = cal_functional("head_plot") # (384,30)

# preprocess -> np.fft.fft


#real part
T_r = torch.load("./dinofeat/functional_map_0_49_True.pt").double()
T_i = torch.load("./dinofeat/functional_map_0_49_False.pt").double()
print(T_r.shape)
T = torch.complex(T_r,T_i)
feat_0 = torch.from_numpy(feat_0)
'''
feat_0_r = torch.matmul(T_r,feat_0.real.double())
feat_0_i = torch.matmul(T_i,feat_0.imag.double())  # 384,256
feat_0 = torch.complex(feat_0_r,feat_0_i).detach().numpy()
'''
feat_0 = torch.matmul(T,feat_0).detach().numpy() # 384,256
feat_0 = feat_0.transpose(1,0) # 256,30
print(feat_0.shape)
feat_0_ifft = np.fft.ifft(feat_0, n=384) # do ifft in the last axis -> 256,384
print(T)
feat_0_ifft = np.abs(feat_0_ifft)


feat_0_ifft = feat_0_ifft.transpose(0,1).reshape((1,256,384))
feat_0_ifft = torch.from_numpy(feat_0_ifft)
torch.save(feat_0_ifft, "./dinofeat/head_plot_ifft.pt")
print(feat_0_ifft.nonzero())
pca_nocorespondence("./dinofeat/head_plot_ifft.pt")
print("done")
'''
if __name__ == "__main__":
    feat_0 = torch.load("./dinofeat/head_plot.pt")
    print(feat_0.shape)
    feat_0_numpy = feat_0.detach().numpy()
    
    feat_0_numpy = feat_0_numpy.transpose(1,2,0)
    print(feat_0_numpy.shape)
  
    fft_feat_0 = np.fft.fft2(feat_0_numpy)
    ifft_feat_0 = np.fft.ifft(fft_feat_0, axis = 1) # (256,384,1)
    ifft_feat_0 = ifft_feat_0.real[:,:,0]
    ifft_feat_0 = ifft_feat_0.transpose(0,1).reshape((1,256,384))
    ifft_feat_0 = torch.from_numpy(ifft_feat_0)
    print(ifft_feat_0.nonzero())
    torch.save(ifft_feat_0, "./dinofeat/head_plot_ifft.pt")
    pca_nocorespondence("./dinofeat/head_plot_ifft.pt")
'''