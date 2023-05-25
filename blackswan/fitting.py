import numpy as np
'''
f1 = np.load("1.pt.npy")
print(f1.shape)
f1_reshape = f1.reshape((-1,1))
f0 = np.load("0.pt.npy")
f0_reshape = f0.reshape((-1,1))
# T x0 = x1     x1 (768,1)   x0-1 (1,768)
print(np.linalg.pinv(f0_reshape).shape)
T = f1_reshape.dot(np.linalg.pinv(f0_reshape))
'''
# use the least squre method to fit a T to 8 features
# first load the features
for i in range(8):
    exec("f"+str(i)+" = np.load('"+str(i)+".pt.npy')")
#then reshape the features
for i in range(8):
    exec("f"+str(i)+"_reshape = f"+str(i)+".reshape((1,-1))")
A = np.vstack((f0_reshape, f1_reshape, f2_reshape, f3_reshape, f4_reshape, f5_reshape, f6_reshape))

B = np.vstack((f1_reshape, f2_reshape, f3_reshape, f4_reshape, f5_reshape, f6_reshape, f7_reshape))

T, residuals, _, _ = np.linalg.lstsq(A, B, rcond=None)



picked = np.zeros((768,1))
picked[0] = 1
next_picked = T.dot(picked)
visualize_f = (next_picked - next_picked.min()) / (next_picked.max() - next_picked.min())
visualize_f = visualize_f * 255
visualize_f = visualize_f.reshape(16, 16, 3).astype(np.uint8)
import matplotlib.pyplot as plt
plt.imshow(visualize_f)
plt.savefig("visualize_f.png")
visualize_init = (picked - picked.min()) / (picked.max() - picked.min())
visualize_init = visualize_init * 255
visualize_init = visualize_init.reshape(16, 16, 3).astype(np.uint8)
plt.imshow(visualize_init)
plt.savefig("visualize_init.png")


if __name__ == "__main__":
    swan_head = np.zeros((16,16,3))
    swan_head[4][8]= 1
    swan_head = swan_head.reshape((-1,1))
    swan_heawd_pca = (swan_head - swan_head.min()) / (swan_head.max() - swan_head.min())
    swan_heawd_pca = swan_heawd_pca * 255
    swan_heawd_pca = swan_heawd_pca.reshape(16, 16, 3).astype(np.uint8)
    plt.imshow(swan_heawd_pca)
    plt.savefig("swan_head.png")
    swan_next_head = T.dot(swan_head.reshape((-1,1)))
    swan_next_head_pca = (swan_next_head - swan_next_head.min()) / (swan_next_head.max() - swan_next_head.min())
    swan_next_head_pca = swan_next_head_pca * 255
    swan_next_head_pca = swan_next_head_pca.reshape(16, 16, 3).astype(np.uint8)
    plt.imshow(swan_next_head_pca)
    plt.savefig("swan_next_head.png")