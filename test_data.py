import numpy as np
data = np.load('data.npy')

var = np.var(data, axis=0)



mean = np.mean(data,axis = 0)



#use heatmap to show the variance of each element in the matrix
import matplotlib.pyplot as plt
import seaborn as sns



plt.figure(figsize=(10, 8))
# use heatmap to show the variance of each element in the matrix
sns.heatmap(var[:,range(9)],vmin = 0, vmax = np.max(var[:,range(9)]))
#save the imag
plt.savefig("var.png")
#use heatmap to show the mean of each element in the matrix
sns.heatmap(mean)
plt.savefig("mean.png")

if __name__ =="__main__":
    from visualizer import visualize
    data_1 = data[0]
    visualize(data_1[:,9],data_1[:,:9],"data1.mp4")
