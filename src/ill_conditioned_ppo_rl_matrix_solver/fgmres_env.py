import gymnasium as gym
from gymnasium import spaces
import numpy as np
from fgmres_solver import fgmres_solver  # Make sure this is the correct import!
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import aslinearoperator

class FGMRESEnv(gym.Env):
    def __init__(self, A, b, ppo_agent=None, max_iters=10, tol=1e-6, restart=5):
        super().__init__()
        self.A_mat = A
        self.A_linop = aslinearoperator(A)
        self.ppo_agent = ppo_agent
        self.b = b
        self.n = A.shape[0]
        self.max_iters = max_iters
        self.tol = tol
        self.restart = restart
        self.iter_count = 0
        self.initial_residual_norm = 0.0
        self.x = np.zeros_like(b)
        self.r = b - self.A_linop @ self.x

        # Define action space: 8 discrete actions for block sizes
        self.action_space = self.action_space = spaces.Discrete(8)
        # Define observation space: 4 continuous features
        # This gives the agent more context for generalization.
        # [0]: Relative residual norm (norm_k / norm_0)
        # [1]: Log of the relative residual norm
        # [2]: Matrix size normalized by log scale (log(n) / log(max_n_seen_in_training))
        # [3]: Current iteration number normalized
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 0.0,  np.inf, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )
    


    def _get_obs(self):
        """Creates the observation vector from the current state."""
        current_residual_norm = np.linalg.norm(self.r)

        if self.initial_residual_norm == 0.0:
            relative_residual_norm = 1.0
        else:
            # This prevents floating-point errors from making the value > 1.0 at reset.
            relative_residual_norm = np.clip(
                current_residual_norm / self.initial_residual_norm, a_min=None, a_max=1.0)

        # Normalize matrix size and iteration count
        normalized_n = np.log10(self.n) / np.log10(4008)
        normalized_iter = self.iter_count / self.max_iters

        # Return a fixed-size array of features
        return np.array([
            relative_residual_norm,
            min(np.log10(relative_residual_norm + 1e-10), 0),  # Add epsilon to avoid log(0)
            normalized_n,
            normalized_iter
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = np.zeros_like(self.b)
        self.r = self.b - self.A_linop @ self.x
        self.iter_count = 0
        self.initial_residual_norm = np.linalg.norm(self.r)

        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        block_size = get_block_size(action, self.n)

        prev_residual_norm = np.linalg.norm(self.r)
        # print(f"[FGMRES] Iter {self.iter_count} | Action: {action} | Block size: {block_size}")
        # Run one outer iteration
        self.x, res_norms, residuals, actions = fgmres_solver(
            self.A_mat.toarray(), self.b, self.x, ppo_agent=self.ppo_agent,
            tol=self.tol, max_iters=1, restart=self.restart, block_size=block_size
        )
        self.r = self.b - self.A_linop @ self.x
        curr_residual_norm = np.linalg.norm(self.r)
        self.iter_count += 1

        terminated = curr_residual_norm < self.tol
        truncated = self.iter_count >= self.max_iters
        # if truncated:
        #     print("Could not converge within max iterations.")
        reward = (prev_residual_norm - curr_residual_norm) / prev_residual_norm
        #print(f"[FGMRES] Iter {self.iter_count} | Residual norm: {curr_residual_norm:.2e} | Reward: {reward:.2e}")

        # obs = self.r.astype(np.float32)
        obs = self._get_obs()

        info = {
            "block_size": block_size,
            "residual_norm": curr_residual_norm,
            "iteration": self.iter_count,
            "action": action,
            "prev_residual_norm": prev_residual_norm,
            "reward": reward
        }

        # Check for divergence (NaN or Inf)
        if not np.isfinite(curr_residual_norm):
            reward = -100.0  # Consistent, large negative reward
            done = True
            info = {"status": "diverged"}
            return self.state, reward, done, info

        # Check for convergence
        if curr_residual_norm <= self.tol:
            reward = 100.0  # Large positive reward for success
            done = True
            info = {"status": "converged"}
            return self.state, reward, done, info

        return obs, reward, bool(terminated), bool(truncated), info

    def render(self, mode="console"):
        if mode != "console":
            raise NotImplementedError()
        #print(f"[FGMRES] Iter {self.iter_count} | Residual norm: {np.linalg.norm(self.r):.2e}")

    def close(self):
        pass


def get_block_size(action, n: int) -> int:
    action = int(np.squeeze(action))
    """
    Map discrete action (0–7) to a block size that adapts with matrix size n.
    
    - Small n → minimum block size floor
    - Medium n → grow slowly with sqrt(n)
    - Large n → capped to avoid huge factorization cost
    """

    # Define scaling factors relative to sqrt(n)
    factors = [0.25, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

    # Compute baseline block size
    base_size = int(factors[action] * np.sqrt(n))

    # Enforce reasonable bounds
    block_size = max(4, base_size)       # minimum useful size
    block_size = min(block_size, 128)    # hard cap to keep runtime reasonable

    return block_size