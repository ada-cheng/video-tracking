import torch
data_0 = torch.load("./dinofeat/00000.pt")
data_0 = data_0.reshape((16,16,384))

import numpy as np

# 执行二维傅里叶变换
def cal_functional(data):
    transformed_features = np.fft.fft2(data)
    magnitude_spectrum = np.log(np.abs(transformed_features))
    sorted_indices = np.argsort(magnitude_spectrum.ravel())[::-1][:30]
    mask = np.zeros_like(magnitude_spectrum, dtype=bool)
    mask[np.unravel_index(sorted_indices, magnitude_spectrum.shape)] = True
    reduced_features = transformed_features * mask
    non_zero = np.count_nonzero(reduced_features)
    non_zero_cols = np.any(reduced_features, axis=0)
    reduced_features = reduced_features[:, non_zero_cols]
    return reduced_features


data_0 = data_0.permute(2,0,1) # 384 16 16

transformed_features = np.fft.fft2(data_0) # 384 16 16 

transformed_features = transformed_features[:,:5,:5] # 384 5 5

transformed_features = transformed_features.reshape((384,-1)) # 384 25


def cal_functional(data_name):
    data = torch.load("./dinofeat/{}.pt".format(data_name)) # f“{data}”.pt"
    data = data.permute(2,0,1)
    transformed_features = np.fft.fft2(data)
   
    transformed_features = transformed_features[:,0,:30]
   
    return transformed_features

data_0 = cal_functional("00000")
data_1 = cal_functional("00001")
print(data_1.shape)
# least square 
# visualize mapping 
# pick a point and visualize mapping



  
# 16 16 30 16 16 30      
'''
magnitude_spectrum = np.log(np.abs(transformed_features))
sorted_indices = np.argsort(magnitude_spectrum.ravel())[::-1][:30]
print(len(sorted_indices))
print(sorted_indices)
'''



'''
mask = np.zeros_like(magnitude_spectrum, dtype=bool)
mask[np.unravel_index(sorted_indices, magnitude_spectrum.shape)] = True
print(np.unravel_index(sorted_indices, magnitude_spectrum.shape))
reduced_features = transformed_features * mask
zero_cols =[]
for i in range(reduced_features.shape[2]):
    if np.count_nonzero(reduced_features[:,:,i]) == 0:
        zero_cols.append(i)
        
reduced_features = np.delete(reduced_features,zero_cols,axis=2)
print(reduced_features.shape)
# 执行二维傅里叶逆变换
inv_features = np.fft.ifft2(reduced_features)
'''