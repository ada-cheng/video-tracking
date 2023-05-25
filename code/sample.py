import torch
from model_mlp import *
from diffusion import *
import numpy as np
model = NoiseMLP(90, 1000, 90)
# load the parameters of the model from model_4.pt
model.load_state_dict(torch.load("model_4.pt"))

# sample 10 matices from the model
samples = sample(model, 90, 10, 1)
#save the samples
np.save ('samples.npy',samples)
print(samples[0].shape)
print(len(samples))
if __name__ == "__main__":
    data = np.load('samples.npy')
    sample1 = data[-1][1]
    T = sample1[:,9]
    for i in range(9):
        print(T.sum(sample1[i]))
    

