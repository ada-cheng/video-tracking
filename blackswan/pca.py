import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import os

'''
def pca(feat):
    feature = torch.load(feat)[0].reshape((256,384))
    pca = PCA(n_components=3)
    pca.fit(feature)
    pca_features=pca.fit_transform(feature)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    plt.imshow(pca_features.reshape(16, 16, 3).astype(np.uint8))
    plt.savefig(feat+".png")
    np.save(feat+".npy",pca_features)
    print(pca_features.shape)
for i in range(8):
    pca(f"./{i}.pt")
'''
f0 = torch.load("./dinofeat/00000.pt")[0].reshape((256, 384))
pca = PCA(n_components=3)
pca.fit(f0)
print(pca.components_.shape)
the_first_three_pcs = pca.components_[0:3, :]
pca_features = pca.transform(f0)
pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
pca_features = pca_features * 255
plt.imshow(pca_features.reshape(16, 16, 3).astype(np.uint8))
plt.savefig("./gendata/0.png")

# use the same pca components to transform another image
color = []
def pcaa(feat, pca):
    # use the pca generated from the first image to transform the other images
    feature = torch.load(f"./gendata/{feat}.pt").reshape((256, 384))
    feature = feature.detach().numpy()
    
    # Set the components to the saved ones
    pca.components_ = the_first_three_pcs

    pca_features = pca.transform(feature)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())# normalize
    
    pca_features = pca_features * 255
    plt.imshow(pca_features.reshape(16, 16, 3).astype(np.uint8))
    color.append(pca_features.reshape(16,16,3)[0][0])
    
    plt.savefig(f"./gendata/{feat}.png")
    np.save(f"./genpca/{feat}.npy", pca_features)

for i in range(49):
    pcaa(i + 1, pca)
plt.plot(color)
plt.savefig("color.png")