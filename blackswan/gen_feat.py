import torch
T = torch.load("./dinofeat/T.pt")
f0 = torch.load("./dinofeat/00000.pt")[0]
for i in range(49):
    exec("f"+str(i+1)+" = torch.matmul(T,f"+str(i)+")")
    exec("torch.save(f"+str(i+1)+",'./gendata/"+str(i+1)+".pt')")