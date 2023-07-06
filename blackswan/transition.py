import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch

for i in range(10):
    exec("feature"+str(i)+" = torch.load('"+str(i)+".pt')")
    

def cal_transition(feature1,feature2):
    # calculate thr transition matrix with the first two features using pesudo inverse
    T_12 = np.array(feature2).dot(np.linalg.pinv(feature1))
    return T_12

for i in range(9):
    exec("T_"+str(i)+"_"+str(i+1)+" = cal_transition(feature"+str(i)+",feature"+str(i+1)+")")

