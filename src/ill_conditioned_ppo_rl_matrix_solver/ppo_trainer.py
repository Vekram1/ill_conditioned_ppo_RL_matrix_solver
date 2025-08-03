# ppo_agent.py

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from fgmres_env import FGMRESEnv

from fgmres_env import get_block_size


class PPOTrainer:
    def __init__(self, A, b, total_timesteps=20000, n_envs=1):
        self.env = make_vec_env(lambda: FGMRESEnv(A, b), n_envs=n_envs)
        self.model = PPO("MlpPolicy", self.env, verbose=1, n_steps=256)
        self.total_timesteps = total_timesteps
        self.actions_taken = []

    def train(self):
        self.model.learn(total_timesteps=self.total_timesteps)

    def save(self, path="ppo_fgmres.zip"):
        self.model.save(path)

    def load(self, path="ppo_fgmres.zip"):
        self.model = PPO.load(path, env=self.env)

    def evaluate(self, episodes=5):
        print(f"Starting evaluation for {episodes} episodes...")
        for episode in range(episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        print("Evaluation completed.")

    def select_action(self, state, n):
        """Use this to get an action in your fgmres_solver loop (during evaluation)."""
        # SB3 expects batched input for prediction
        action, _ = self.model.predict(state.reshape(1, -1), deterministic=True)
        self.store_action(action)
        """return  block size """
        block_size = get_block_size(action, n)
        return block_size
    
    def store_action(self, action):
        """Call this during evaluation or solving to log the chosen action."""
        if hasattr(action, "item"):
            self.actions_taken.append(action.item())
        else:
            self.actions_taken.append(action)

    def get_actions(self):
        """Returns the list of all stored actions (e.g., after solve or evaluation)."""
        return self.actions_taken

    def clear_actions(self):
        """Clear the stored actions (optional utility)."""
        self.actions_taken.clear()
