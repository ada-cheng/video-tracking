import torch
import numpy as np
import matplotlib.pyplot as plt
# track one point
p1 = np.zeros((16,16,1))
p1[4][8] = 1
plt.imshow(p1)
plt.axis('off')
plt.savefig("./trajectory/p1.png",dpi = 256, bbox_inches='tight',pad_inches=0)
p1 = p1.reshape((-1,1))
T  = torch.load("./dinofeat/T.pt")
# change T to numpy
T = T.detach().numpy()
p2 = T.dot(p1)
p2 = p2.reshape((16,16,1))


plt.imshow(p2)
plt.axis('off')
plt.savefig("./trajectory/p2.png",dpi = 256, bbox_inches='tight',pad_inches=0)

'''
for i in range(48):
    p2 = T.dot(p2.reshape((-1,1)))
    p2 = p2.reshape((16,16,1))
    plt.imshow(p2)
    plt.axis('off')
    plt.savefig("./trajectory/p"+str(i+3)+".png",dpi = 256, bbox_inches='tight',pad_inches=0)
'''

'''
import os
import random
for i in range(5):
    directory = "./trajectory/"+str(i)
    if not os.path.exists(directory):
        os.makedirs(directory)
    p1 = np.zeros((16,16,1))
    p1[random.randint(0,15)][random.randint(0,15)] = 1
    plt.imshow(p1)
    plt.axis('off')
    plt.savefig(directory+"/p1.png")
    p1 = p1.reshape((-1,1))
    T  = torch.load("./dinofeat/T.pt")
    # change T to numpy
    T = T.detach().numpy()
    for i in range(48):
        p2 = T.dot(p1.reshape((-1,1)))
        p2 = p2.reshape((16,16,1))
        plt.imshow(p2)
        plt.axis('off')
        plt.savefig(directory+"/p"+str(i+2)+".png",dpi = 256, bbox_inches='tight',pad_inches=0)
        p1 = p2
'''