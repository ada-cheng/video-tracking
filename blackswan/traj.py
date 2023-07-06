from PIL import Image

def overlay_images(image_top_path, image_bottom_path, opacity):
    # 打开上方图片和下方图片
    image_top = Image.open(image_top_path).convert("RGBA")
    image_bottom = Image.open(image_bottom_path).convert("RGBA")

    
    # 调整下方图片为灰色
    image_bottom_gray = image_bottom.convert("L").convert("RGBA")

    # 调整上方图片的透明度
    image_top = image_top.resize((854,480))
  
    
    image_top = Image.blend(image_top, image_bottom_gray, opacity)

    # 将上方图片叠加到下方图片上
    image_result = Image.alpha_composite(image_bottom, image_top)

    # 保存结果图片
    image_result.save("./cover/"+image_bottom_path[2:-4]+".png")





if __name__ == "__main__":
    '''
    top = Image.open("./trajectory/p5.png").convert("RGBA")
    #resize the image to (854,480)
    top = top.resize((854,480))
    bottom = Image.open("./00000.jpg").convert("RGBA")
    image_top = Image.blend(top, bottom, 0.5)
    image_top.save("./cover/00000.png")
    '''
    



    
    for i in range(0,50):
        i = str(i).zfill(5)
        overlay_images("./trajectory/p"+str(int(i)+1)+".png", "./"+str(i)+".jpg", 0.5)
    