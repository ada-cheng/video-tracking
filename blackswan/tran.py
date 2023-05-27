'''

import torch
f0 = torch.load("./dinofeat/00000.pt")[0]
f1 = torch.load("./dinofeat/00001.pt")[0]
# T f0 = f1 (T.shape(256,256),f0.shape(256,384),f1.shape(256,384))
T = f1.matmul(torch.pinverse(f0))


reshape_0 = torch.reshape(f0,(1,-1)) 
reshape_1 = torch.reshape(f1,(1,-1))
print(reshape_0.shape)
T_ = torch.pinverse(reshape_0).matmul(reshape_1)
print(T_.shape)
'''
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

# Define the matrices M1 to M9
M1 = torch.load("./dinofeat/00000.pt")[0]
M2 = torch.load("./dinofeat/00001.pt")[0]
M3 = torch.load("./dinofeat/00002.pt")[0]
M4 = torch.load("./dinofeat/00003.pt")[0]
M5 = torch.load("./dinofeat/00004.pt")[0]
M6 = torch.load("./dinofeat/00005.pt")[0]
M7 = torch.load("./dinofeat/00006.pt")[0]
M8 = torch.load("./dinofeat/00007.pt")[0]
M9 = torch.load("./dinofeat/00008.pt")[0]


# Define the T matrix as a trainable parameter
T = nn.Parameter(torch.randn(256, 256))

# Define the loss function (mean squared error)
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam([T], lr=0.01)

# Perform the optimization
for epoch in range(10000):
    optimizer.zero_grad()

    # Compute the predicted matrices
    M2_pred = torch.matmul(T, M1)
    M3_pred = torch.matmul(T, M2)
    M4_pred = torch.matmul(T, M3)
    M5_pred = torch.matmul(T, M4)
    M6_pred = torch.matmul(T, M5)
    M7_pred = torch.matmul(T, M6)
    M8_pred = torch.matmul(T, M7)
    M9_pred = torch.matmul(T, M8)

    # Compute the loss
    loss = loss_fn(M2_pred, M2) + loss_fn(M3_pred, M3) + loss_fn(M4_pred, M4) + \
           loss_fn(M5_pred, M5) + loss_fn(M6_pred, M6) + loss_fn(M7_pred, M7) + \
           loss_fn(M8_pred, M8) + loss_fn(M9_pred, M9)

    # Backpropagation and optimization step
    loss.backward()
    optimizer.step()

    # Print the loss during training
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}")

# Print the learned T matrix
print("Learned T matrix:")
print(T)
torch.save(T,"./dinofeat/T.pt")
