from diffusion import *
import numpy as np
from model_mlp import NoiseMLP
from torch.optim import Adam
import matplotlib.pyplot as plt

#first load data from data.npy
data = np.load('data.npy')

device = "cuda" if torch.cuda.is_available() else "cpu"
if  not torch.cuda.is_available():
    print("not Using the GPU!")
model = NoiseMLP(90, 1000, 90).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
#reshape data to a 2d tensor
data = data.reshape(data.shape[0], -1)

losses = []


# start training
timesteps = 1000
epochs = 5
for epoch in range(epochs):
    for step, batch in enumerate(data): 
        optimizer.zero_grad()
        batch = torch.from_numpy(batch).float().to(device)# each batch represents a image like sth
        t = torch.randint(0, timesteps, (batch.shape[0],)).to(device)#t.shape = (batch_size,) t[i] means the time step of the ith image
        loss = p_losses(model, batch, t,loss_type="huber")
        losses.append(loss.item())
        
        if step % 100 == 0:
            print(f"epoch: {epoch}, step: {step}, loss: {loss.item()}")
            
        loss.backward()    
        optimizer.step()
    torch.save(model.state_dict(), f"model_{epoch}.pt")
    print(f"model saved at epoch {epoch}")
    plt.plot(losses)
    plt.title(f"loss_{epoch}")
    plt.xlabel("step") #step means the number of images
    plt.ylabel("loss")
    plt.savefig(f"loss_{epoch}.png")
  




    
    
    
    

     
