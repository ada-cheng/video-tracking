import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import os


def pca_nocorespondence(feat,feat_dim):
   
    feature = torch.load(feat)[0].reshape((256,feat_dim))
    
    pca = PCA(n_components=3)
    pca.fit(feature)
  
    pca_features=pca.fit_transform(feature)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())
    pca_features = pca_features * 255
    plt.imshow(pca_features.reshape(16, 16, 3).astype(np.uint8))
    if feat.startswith("./gendata/"):
        feat = feat.lstrip("./gendata/").split(".")[0]
    if feat.startswith("./dinofeat/"):
        feat = feat.lstrip("./dinofeat/").split(".")[0]
 
    plt.savefig(f"./gendata/{feat}.png")
    np.save(f"./genpca/{feat}.npy", pca_features)
    


def pcaa(feat, feat_dim,previous_pca):
    # use the pca generated from the first image to transform the other images
    if isinstance(feat,int):
        feature = torch.load(f"./gendata/{feat}.pt").reshape((256, feat_dim))
    else:
        feature = torch.load(feat)[0].reshape((256, feat_dim))
    feature = feature.detach().numpy()
    
    # Set the components to the saved ones
    pca = previous_pca



    pca_features = pca.transform(feature)
    pca_features = (pca_features - pca_features.min()) / (pca_features.max() - pca_features.min())# normalize
    
    pca_features = pca_features * 255
    plt.imshow(pca_features.reshape(16, 16, 3).astype(np.uint8))
 
    if isinstance(feat, str):
        if feat.startswith("./gendata/"):
            feat = feat.lstrip("./gendata/").split(".")[0]
        if feat.startswith("./dinofeat/"):
            feat = feat.lstrip("./dinofeat/").split(".")[0]
    plt.savefig(f"./gendata/{feat}.png")
    np.save(f"./genpca/{feat}.npy", pca_features)
if __name__ == "__main__":

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
  
    for i in range(49):
        pcaa(i + 1, 384,pca)
 