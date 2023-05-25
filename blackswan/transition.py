import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch

ft0 = np.load("0.pt.npy")
ft1 = np.load("1.pt.npy")
# calculate the inverse of ft0 using pseudo inverse
ft0_inv = np.linalg.pinv(ft0.reshape(-1,1))# linalg means linear algebra

print(ft0_inv.shape)

T = ft1.reshape((-1,1)).dot(ft0_inv)
# visualize the transformation matrix
plt.imshow(T,cmap="gray") 
plt.savefig("T.png")