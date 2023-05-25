import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch


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
    