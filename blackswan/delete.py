import os

# 指定目录路径
directory = '.'

# 获取目录下的所有文件
file_list = os.listdir(directory)

# 遍历文件列表并删除文件
for file_name in file_list:
    # 构建文件的完整路径
    file_path = os.path.join(directory, file_name)
    
    # 判断文件是否存在并且是否为文件
    if os.path.isfile(file_path):
        # 删除文件
        if file_path.endswith(".npy"):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
