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
        #self.load_matrix_A(group="Grund", name="poli")
        self.group = None
        self.name = None

    def load_matrix_A(self, group="Grund", name="poli"):
        # Download the file
        self.group = group
        self.name = name
        print(self.group, self.name)
        filepath_mat = f"{name}.mat"
        if not os.path.exists(filepath_mat):
            base_url = f"https://suitesparse-collection-website.herokuapp.com/mat/{group}/{name}.mat"
            print(f"Downloading {filepath_mat} from {base_url}")
            response = requests.get(base_url)
            with open(filepath_mat, 'wb') as f:
                f.write(response.content)
                print(f"Downloaded {filepath_mat}")

        # Try loading the file. Some matrices in this collection are .mtx, not .mat.
        try:
            # First, try to load as a .mat file
            mat = loadmat(filepath_mat)
            problem = mat['Problem'][0, 0]
            A = problem['A']
            print(f"Successfully loaded {filepath_mat} as a .mat file.")
        except (FileNotFoundError, KeyError):
            # If that fails, assume it's a Matrix Market format and try to read it.
            # This will handle the string header.
            filepath_mtx = f"{name}.mtx"
            if not os.path.exists(filepath_mtx):
                print(f"{filepath_mat} failed to load, trying to download and load as .mtx")
                base_url = f"https://suitesparse-collection-website.herokuapp.com/mat/{group}/{name}.mtx"
                response = requests.get(base_url)
                with open(filepath_mtx, 'wb') as f:
                    f.write(response.content)
                    print(f"Downloaded {filepath_mtx}")

            A = mmread(filepath_mtx)
            print(f"Successfully loaded {filepath_mtx} as a Matrix Market file.")

        # Convert the matrix to CSR format and set the class variable
        self.matrix = csr_matrix(A.astype(np.float64))
        print(f"Matrix A loaded with dimensions: {self.matrix.shape}")
        
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
        A_group = A.group
        A_name = A.name
        self.vector = None
        if (A.get_matrix() is not None):
            #self.get_b(A_group, A_name, A)
            self.generate_random_b(A.get_shape())
        else:
            self.vector = None

    def generate_random_b(self, n):
        self.vector = np.random.randn(n)
    
    def get_b(self, group, name, A=None):
        filepath_mat_b = f"{name}_b.mat"
        if not os.path.exists(filepath_mat_b):
            base_url = f"https://suitesparse-collection-website.herokuapp.com/mat/{group}/{name}.mat"
            response = requests.get(base_url)
            with open(filepath_mat_b, 'wb') as f:
                f.write(response.content)
                print(f"Downloaded {filepath_mat_b}")
                mat = loadmat(filepath_mat_b)
                problem = mat['Problem'][0, 0]
                try:
                    b = problem['b']
                    print(f"Successfully loaded {filepath_mat_b} as a .mat file.")
                except ValueError:
                    os.remove(filepath_mat_b)
                    self.generate_random_b(A.get_shape())
                    return
        self.vector = b.flatten().astype(np.float64)




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
