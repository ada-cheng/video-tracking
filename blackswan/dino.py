import torch
from torchvision.transforms import functional as F
from PIL import Image
import os

# 加载预训练模型
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
print("model\n")
# 加载并预处理自定义图片
def get_feature(image_path,output):
    image = Image.open(image_path)
    # reshape the image
    image = image.resize((224, 224))
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    with torch.no_grad():
        features = model.get_intermediate_layers(image.unsqueeze(0), 1, return_class_token=True)[0] 
    
    # save to the dinofeat folder
    torch.save(features[0],os.path.join("./dinofeat",output))
for i in range(50):
    # the name is zero filled to 5 digits
    i = str(i).zfill(5)
    get_feature("./"+str(i)+".jpg",str(i)+".pt")



