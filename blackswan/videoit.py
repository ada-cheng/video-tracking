import cv2
import os

def images_to_video(image_folder, output_path, fps):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    

    # Sort the image files numerically
    image_files.sort(key=lambda x: int(x.split('.')[0]))

    # Determine the width and height of the images
    sample_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = sample_image.shape
   
    # Create a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use appropriate codec based on the file extension
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Iterate through each image file and write it to the video
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
       
        image = cv2.imread(image_path)
        video_writer.write(image)
        

    # Release the video writer and close the video file
    video_writer.release()
    cv2.destroyAllWindows()

# Example usage
image_folder = '.'
output_path = './output.mp4'
fps = 24  # Frames per second

images_to_video(image_folder, output_path, fps)


