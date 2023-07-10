import torch
from matplotlib import pyplot as plt
head = torch.zeros((16,16,384))
head[4][8][0] = 1
head_npy = head.detach().numpy()
np.save("./dinofeat/head.npy",head_npy)
