import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from pca import pcaa, pca_nocorespondence

#eigen_dim
dim1 = dim2 = 5
feat_dim = 384



# get the functional map F0
from fft import cal_functional


#first we get the main pca_components of f0
f0 = torch.load("./dinofeat/00000.pt")[0].reshape((256, 384))
pca = PCA(n_components=3)
pca.fit(f0)
the_first_three_pcs = pca.components_[0:3, :]
np.save("./genpca/the_first_three_pcs.npy", the_first_three_pcs)
pca_features = pca.transform(f0)
pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
pca_features = pca_features * 255
plt.imshow(pca_features.reshape(16, 16, 3).astype(np.uint8))
plt.savefig("./gendata/0.png")

feat_0 = cal_functional("00000",dim1,dim2) # (5,5,384)





feat_0 = feat_0.reshape((dim1*dim2,-1)) # (25,384)

# get the T_0_49
#real part and imaginary part
T_r = torch.load("./dinofeat/functional_map_0_49_True.pt").double()
T_i = torch.load("./dinofeat/functional_map_0_49_False.pt").double()
print(T_r.shape)
T = torch.complex(T_r,T_i) # (25,25)
feat_0 = torch.from_numpy(feat_0)

# transition to the new space get F_49
feat_0 = torch.matmul(T,feat_0).detach().numpy() # 25,384
feat_0 = feat_0.transpose(1,0) # 384,25
print(feat_0.shape)
feat_0 = feat_0.reshape((384,dim1,dim2)) # 384,5,5
feat_0 = np.pad(feat_0,((0,0),(0,16 - dim1),(0,16 - dim2))) # padding inorder to get the ifft
feat_0_ifft = np.fft.ifft2(feat_0) # 384,16,16
print(T)
feat_0_ifft = np.abs(feat_0_ifft)
print(feat_0_ifft.shape)

# pca visualization
feat_0_ifft = feat_0_ifft.reshape((384,-1)).transpose(1,0)#  256 384
feat_0_ifft = torch.from_numpy(feat_0_ifft)[None,:,:] # 1 256 384
torch.save(feat_0_ifft, "./dinofeat/0->49.pt")
print(feat_0_ifft.nonzero())
pca_nocorespondence("./dinofeat/0->49.pt",feat_dim)
print("done")
'''
if __name__ == "__main__":
    feat_0 = torch.load("./dinofeat/head_plot.pt")
    print(feat_0.shape) # 1 256 384
    feat_0_numpy = feat_0.detach().numpy()
    
    feat_0_numpy = feat_0_numpy.transpose(2,1,0)# 384 256 1
    print(feat_0_numpy.shape)
  
    fft_feat_0 = np.fft.fft2(feat_0_numpy)# 384 256 1
    print(fft_feat_0.shape)
    ifft_feat_0 = np.fft.ifft2(fft_feat_0) # 384 256 1
    print(ifft_feat_0.shape)
    

    ifft_feat_0 = ifft_feat_0.transpose(2,1,0).reshape((1,256,384))
    ifft_feat_0 = torch.from_numpy(ifft_feat_0)
    ifft_feat_0 = np.abs(ifft_feat_0)
    print(ifft_feat_0.nonzero())
    torch.save(ifft_feat_0, "./dinofeat/head_plot_ifft.pt")
    pca_nocorespondence("./dinofeat/head_plot_ifft.pt")
'''