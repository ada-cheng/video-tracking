import numpy as np
data0 = np.load('data.npy')[0]
print(np.sum(data0[:,0:9],axis=1))
sample0= np.load('samples.npy')[-1][0].reshape(9,10)
print(np.sum(sample0[:,0:9],axis=1))