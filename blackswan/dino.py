import torch
from torchvision.transforms import functional as F
from PIL import Image

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
    print(features[0].shape)
    torch.save(features[0], output)
for i in range(10):
    get_feature(f"./0000{i}.jpg",f"./{i}.pt")



