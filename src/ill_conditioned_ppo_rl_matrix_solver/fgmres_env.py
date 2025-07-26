import gym
from gym import spaces
import numpy as np
from fgmres_solver import fgmres_step  # You will implement this
from scipy.sparse.linalg import aslinearoperator

from stable_baselines.common.env_checker import check_env


class FGMRESEnv(gym.Env):
    """
    Custom Environment for solving Ax = b with FGMRES using adaptive block QR preconditioning.
    Action: block size for QR preconditioner.
    Observation: residual vector (or norm) after applying FGMRES step.
    """
    metadata = {"render.modes": ["console"]}

    def __init__(self, A, b, max_iters=100, tol=1e-6, min_block=1, max_block=200):
        super(FGMRESEnv, self).__init__()

        self.A = aslinearoperator(A)
        self.b = b
        self.n = A.shape[0]
        self.max_iters = max_iters
        self.tol = tol
        self.iter_count = 0

        max_block = max(max_block, self.n) # the max block size should be the size of the matrix

        # Initial guess
        self.x = np.zeros_like(b)
        self.r = b - self.A @ self.x

        # Action space: continuous block size
        self.action_space = spaces.Box(
            low=np.array([min_block]),
            high=np.array([max_block]),
            dtype=np.float32
        )

        # Observation: residual norm and optionally current iteration
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n,), dtype=np.float32
        )

    def reset(self):
        self.x = np.zeros_like(self.b)
        self.r = self.b - self.A @ self.x
        self.iter_count = 0
        return self.r.astype(np.float32)

    def step(self, action):
        # Clip and round action to valid block size
        block_size = int(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))

        # Apply one FGMRES restart step with this block size
        # self.x, self.r = fgmres_step(self.A, self.b, self.x, block_size)

        residual_norm = np.linalg.norm(self.r)
        self.iter_count += 1

        # Reward is negative residual norm (want to minimize)
        reward = -residual_norm

        # Done if convergence or max iterations
        done = residual_norm < self.tol or self.iter_count >= self.max_iters

        info = {"block_size": block_size, "residual_norm": residual_norm}

        return self.r.astype(np.float32), reward, done, info

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        print(
            f"Iteration: {self.iter_count}, Residual norm: {np.linalg.norm(self.r):.4e}")

    def close(self):
        pass


env = FGMRESEnv()
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)
