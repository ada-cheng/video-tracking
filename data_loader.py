import numpy as np

# Define the number of samples and the dimensions of the matrices
num_samples = 1000
matrix_dim = 9
EPS = 1e-6
# Initialize the arrays to hold the matrices and vectors
matrices = np.zeros((num_samples, matrix_dim, matrix_dim+1))

# Generate the matrices and vectors
for i in range(num_samples):
    # Generate a random matrix and normalize it
    matrix = np.random.rand(matrix_dim, matrix_dim)
    matrix_norm = matrix / (np.sum(matrix, axis=1)+EPS)
    
    # Generate a vector with a single 1 in each column
    x0 = np.zeros((matrix_dim, 1))
    x0[np.random.randint(9)] = 1

    vector = x0
    # Concatenate the matrix and vector
    matrix_concat = np.concatenate((matrix_norm, vector), axis=1)
   
   
    # Add the matrix_concat to the arrays
    matrices[i] = np.concatenate((matrix_norm,vector),axis = 1)
    
   

data = matrices
#delete matrices[i] if matrices[i] = matrices[k] for i != k
for i in range(num_samples):
    for j in range(i+1,num_samples):
        if np.array_equal(data[i],data[j]):
            data = np.delete(data,j,axis = 0)
            num_samples -= 1
            

# Save the data to a .npy file
np.save('data.npy', data)

#load a sample from the file data.npy
data = np.load('data.npy')
print(data.shape)
print(data[0].shape)

