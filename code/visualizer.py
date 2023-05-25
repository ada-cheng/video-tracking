import numpy as np
import cv2
EPS = 1e-6
def visualize(x,  T,name):
    
    # Define the number of frames and the video size
    num_frames = 20
    video_size = (300, 300)

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(name, fourcc, 10.0, video_size)

    # Generate the video frames
    for i in range(num_frames):
        # Compute the current state xt from the previous state xtm1 and the transition matrix T
        if i == 0:
            xt = x
        else:
            xt = T.dot(xtm1)
        xtm1 = xt

        # Convert the state vector to a binary matrix and reshape it to a 3x3 matrix
        binary_matrix = xt.reshape((3, 3))
    
        # Scale up the binary matrix using nearest neighbor interpolation and convert it to a grayscale image
        scaled_matrix = cv2.resize(binary_matrix, video_size, interpolation=cv2.INTER_NEAREST)
  
        image = np.uint8(scaled_matrix * 255)
  
        # Add the image to the video writer
        out.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))

    # Release the video writer and display the video
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # load the data from samples.npy
    samples = np.load('samples.npy')
    # get the first sample's last stage
    x = samples[-1][5]
    # reshape the sample to a 9*10
    x = x.reshape(9,10)
    # T is the first 9 columns
    T = x[:,:9]
    
    x = x[:,9]
    visualize(x,T,"diffusion_output.mp4")
    