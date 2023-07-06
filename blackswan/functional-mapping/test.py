import numpy as np

def compute_projection(f, basis_size):
    projection = np.zeros(basis_size)  # 初始化投影数组
    height, width, embedding_dim = f.shape
    x_vals = np.arange(height)[:, np.newaxis, np.newaxis]  # 生成 x 值
    y_vals = np.arange(width)[np.newaxis, :, np.newaxis]  # 生成 y 值
    basis = np.zeros((height, width, basis_size, embedding_dim))  # 初始化基函数数组
    
    # 计算基函数
    for i in range(basis_size):
        basis[:, :, i, :] = np.sin(i * x_vals / height * np.pi) * np.sin(i * y_vals / width * np.pi) # 生成基函数 = sin(i * x / height * pi) * sin(i * y / width * pi)
    
    # 计算投影系数
    for i in range(basis_size):
        projection[i] = np.sum(f * basis[:, :, i, :]) # 计算投影系数 = f * 基函数 = f * sin(i * x / height * pi) * sin(i * y / width * pi) 
        
    
    return projection

# 假设 'embedding' 是从 dinov2 模型得到的大小为 (256, 384) 的嵌入向量
embedding = np.random.rand(16,16, 384)  # 示例随机生成嵌入向量
patch_size = 8  # 指定 patch 的大小
x, y = 1, 1  # 指定 patch 的位置

patch_embedding = embedding[x:x+patch_size, y:y+patch_size, :]  # 提取所需的 patch 嵌入向量

# 指定所需的基函数数量（基函数的个数）
basis_size = 10

# 计算 patch 嵌入向量在基函数上的投影
projection = compute_projection(patch_embedding, basis_size)

print(projection)


