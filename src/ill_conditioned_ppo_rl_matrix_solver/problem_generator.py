import requests
import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import loadmat

class matrix_A:
    def __init__(self):
        self.shape = None
        self.sparsity = None
        self.matrix = None
        self.load_matrix_A(filepath="poli.mat")

    def load_matrix_A(self, filepath, name="poli"):
        if not os.path.exists(filepath):
            base_url = f"https://suitesparse-collection-website.herokuapp.com/mat/Grund/poli.mat"
            response = requests.get(base_url)
            with open(f"{name}.mat", 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {name}.mat")
        mat = loadmat(filepath)
        A = mat['Problem'][0][0][1] 
        self.matrix = csr_matrix(A)
        
    def get_sparsity(self):
        if self.matrix is not None:
            return 1.0 - (np.count_nonzero(self.matrix) / (self.matrix.shape[0] * self.matrix.shape[1]))
        else:
            raise ValueError("Matrix A is not loaded yet.")
    
    def get_shape(self):
        if self.matrix is not None:
            return self.matrix[0].shape[1]
        else:
            raise ValueError("Matrix A is not loaded yet.")
    
    def get_matrix(self):
        if self.matrix is not None:
            return self.matrix
        else:
            raise ValueError("Matrix A is not loaded yet.")

class vector_b:
    def __init__(self, A):
        if (A.get_matrix() is not None):
            self.generate_random_b(A.get_shape())
        else:
            self.vector = None

    def generate_random_b(self, n):
        self.vector = np.random.randn(n)



# def generate_sparse_corr_like_ill_conditioned(n, block_size=10, epsilon=1e-4, seed=0):
#     np.random.seed(seed)
#     A = np.eye(n)

#     # Introduce a set of near-linear dependencies in random blocks
#     for i in range(0, n, block_size):
#         idx = slice(i, min(i + block_size, n))
#         block = np.random.randn(block_size, block_size)
#         block = block @ block.T  # make it PSD

#         # Normalize to have 1s on diagonal
#         D = np.sqrt(np.diag(block))
#         block_corr = block / np.outer(D, D)
#         np.fill_diagonal(block_corr, 1)

#         # Make it near-singular
#         noise = epsilon * np.random.randn(*block_corr.shape)
#         noise = (noise + noise.T) / 2  # enforce symmetry
#         np.fill_diagonal(noise, 0)  # no noise on diagonal
#         block_corr += noise

#         # Optional: project to PSD (clip eigenvalues)
#         eigvals, eigvecs = np.linalg.eigh(block_corr)
#         eigvals = np.clip(eigvals, 1e-8, None)
#         block_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T

#         # Insert into the big matrix
#         A[idx, idx] = block_corr[:(idx.stop - idx.start),
#                                  :(idx.stop - idx.start)]

#     return A
