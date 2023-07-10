import torch
from matplotlib import pyplot as plt
T_01_real = torch.load("./functional_map_01_real.pt")
T_01_imag = torch.load("./functional_map_01_imag.pt")
plt.imshow(T_01_imag.detach().numpy())
plt.savefig("./functional_map_01_imag.png")
