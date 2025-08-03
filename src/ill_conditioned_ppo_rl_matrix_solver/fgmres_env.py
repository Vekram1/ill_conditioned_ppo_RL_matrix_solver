import gymnasium as gym
from gymnasium import spaces
import numpy as np
from fgmres_solver import fgmres_solver  # Make sure this is the correct import!
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator

class FGMRESEnv(gym.Env):
    def __init__(self, A, b, max_iters=10, tol=1e-6, restart=5):
        super().__init__()
        self.A_mat = A
        self.A_linop = aslinearoperator(A)
        self.b = b
        self.n = A.shape[0]
        self.max_iters = max_iters
        self.tol = tol
        self.restart = restart
        self.iter_count = 0

        self.x = np.zeros_like(b)
        self.r = b - self.A_linop @ self.x

        #self.block_sizes = [2 ** i for i in range(2, int(np.floor(np.log2(self.n))) + 1)]
        self.action_space = self.action_space = spaces.Discrete(7)
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(self.n,), dtype=np.float32
        # )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = np.zeros_like(self.b)
        self.r = self.b - self.A_linop @ self.x
        self.iter_count = 0
        initial_residual_norm = np.linalg.norm(self.r)
        obs = np.array([initial_residual_norm], dtype=np.float32)
        return obs, {}

    def step(self, action):
        block_size = get_block_size(action, self.n)



        prev_residual_norm = np.linalg.norm(self.r)
        print(f"[FGMRES] Iter {self.iter_count} | Action: {action} | Block size: {block_size}")
        # Run one outer iteration
        self.x, res_norms, residuals, actions = fgmres_solver(
            self.A_mat.toarray(), self.b, ppo_agent=None,
            tol=self.tol, max_iters=1, restart=self.restart, block_size=block_size
        )
        self.r = self.b - self.A_linop @ self.x
        curr_residual_norm = np.linalg.norm(self.r)
        self.iter_count += 1

        terminated = curr_residual_norm < self.tol
        truncated = self.iter_count >= self.max_iters
        reward = prev_residual_norm - curr_residual_norm
        print(f"[FGMRES] Iter {self.iter_count} | Residual norm: {curr_residual_norm:.2e} | Reward: {reward:.2e}")

        # obs = self.r.astype(np.float32)
        obs = np.array([curr_residual_norm], dtype=np.float32)
        info = {
            "block_size": block_size,
            "residual_norm": curr_residual_norm,
            "iteration": self.iter_count,
            "action": action,
            "prev_residual_norm": prev_residual_norm,
            "reward": reward
        }
        return obs, reward, bool(terminated), bool(truncated), info

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        #print(f"[FGMRES] Iter {self.iter_count} | Residual norm: {np.linalg.norm(self.r):.2e}")

    def close(self):
        pass


def get_block_size(action, n):
    if action == 0:
        block_size = min(4, n)  # Smallest absolute block size
    elif action == 1:
        block_size = min(8, n)  # A slightly larger absolute block size
    elif action == 2:
        block_size = min(int(n * .10), n)  # 37.5% of matrix size
    elif action == 3:
        block_size = min(int(n * 0.15), n)  # 50% of matrix size
    elif action == 4:
        block_size = min(int(n * 0.20), n)  # 20% of matrix size
    elif action == 5:
        block_size = min(int(n * 0.25), n)  # 30% of matrix size
    elif action == 6:
        block_size = min(int(n * 0.275), n)  # 40% of matrix size

    return block_size
