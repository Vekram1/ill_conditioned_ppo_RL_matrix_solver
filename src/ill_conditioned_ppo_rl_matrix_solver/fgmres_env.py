import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scipy.sparse.linalg as spla


class FGMRESEnv(gym.Env):
    """
    FGMRES environment for solving Ax = b using block-preconditioned FGMRES.
    Actions: block sizes (discrete or continuous)
    State: residual vector r
    Reward: negative residual norm after applying FGMRES step
    """

    def __init__(self, A, b, max_iterations=100, convergence_threshold=1e-5, min_block_size=1):
        super().__init__()
        self.A = A  # System matrix (assumed dense or sparse matrix)
        self.b = b  # RHS vector
        self.n = A.shape[0]
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        self.min_block_size = min_block_size
        self.max_block_size = self.n

        # Observation: current residual vector (float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n,), dtype=np.float32)

        # Action: block size from min_block_size to max_block_size
        self.action_space = spaces.Discrete(
            self.max_block_size - self.min_block_size + 1)

        # Internal state
        self.iteration = 0
        self.x = np.zeros_like(b)
        self.current_residual_vector = self.b - self.A @ self.x
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.iteration = 0
        self.x = np.zeros_like(self.b)
        self.current_residual_vector = self.b - self.A @ self.x
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        return self.current_residual_vector.astype(np.float32)

    def _calculate_reward(self):
        return -np.linalg.norm(self.current_residual_vector)

    def _apply_block_qr_preconditioner(self, r, block_size):
        """
        Apply block QR preconditioning to the residual vector r.
        This method partitions the matrix A into blocks of size block_size,
        computes the QR decomposition for each block, and applies the preconditioner.
        """
        if block_size < self.min_block_size or block_size > self.max_block_size:
            raise ValueError(f"Block size must be between {self.min_block_size} and {self.max_block_size}")
        n = self.A.shape[0]
        precond_r = np.zeros_like(r)
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            A_block = self.A[start:end, start:end]
            r_block = r[start:end]

            # Ensure block is full-rank; otherwise skip
            if A_block.shape[0] == 0 or A_block.shape[1] == 0:
                continue

            try:
                Q, R = np.linalg.qr(A_block)
                # Solve R y = Qᵀ r
                y = np.linalg.solve(R, Q.T @ r_block)
                precond_r[start:end] = y
            except np.linalg.LinAlgError:
                # Fallback to identity if QR fails
                precond_r[start:end] = r_block

        return precond_r

    def step(self, action):
        block_size = int(action)

        # Compute residual: r = b - A @ x
        r = self.b - self.A @ self.x

        # Apply QR preconditioner
        z = self._apply_block_qr_preconditioner(r, block_size)

        # Use z as search direction: x_new = x + z (simplified 1-step update)
        self.x = self.x + z

        # New residual
        self.current_residual_vector = self.b - self.A @ self.x

        reward = self._calculate_reward()
        done = np.linalg.norm(self.current_residual_vector) < self.convergence_threshold
        obs = self._get_obs()

        return obs, reward, done, {}

    def render(self):
        print(
            f"Step {self.iteration}, Residual Norm: {np.linalg.norm(self.current_residual_vector):.3e}")
