from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from fgmres_env import FGMRESEnv, get_block_size
import numpy as np


class PPOTrainer:
    def __init__(self, n_envs=1):
        """
        Initializes the trainer. The agent model is created in a separate
        method to allow for dynamic environment setup.
        """
        self.model = None
        self.n_envs = n_envs
        self.actions_taken = []

    def setup_env(self, env_list):
        """
        Creates a vectorized environment from a list of environment functions
        and initializes the PPO model.

        Args:
            env_list (list): A list of lambda functions, each creating an FGMRESEnv.
        """
        self.env = make_vec_env(env_list, n_envs=self.n_envs)
        self.model = PPO("MlpPolicy", self.env, verbose=1, n_steps=256)

    def train(self, env, total_timesteps):
        """
        Trains the PPO model on the provided environment.

        Args:
            env: The environment to train on.
            total_timesteps (int): The number of timesteps to train for.
        """
        if self.model is None:
            # If the model has not been initialized yet, create it now
            self.model = PPO("MlpPolicy", env, verbose=1, n_steps=256)
        else:
            # If the model exists, set its new environment
            self.model.set_env(env)

        # Now train the model for the specified timesteps
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path="ppo_fgmres.zip"):
        """Saves the trained model."""
        if self.model:
            self.model.save(path)

    def load(self, path="ppo_fgmres.zip"):
        """Loads a pre-trained model."""
        self.model = PPO.load(path)

    def evaluate(self, env, episodes=5):
        """
        Evaluates the trained model on a specific environment.

        Args:
            env: The environment to evaluate on.
            episodes (int): The number of evaluation episodes.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call setup_agent() or load() first.")

        print(f"Starting evaluation for {episodes} episodes...")
        for episode in range(episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        print("Evaluation completed.")

    def select_action(self, state, n):
        """Use this to get an action from the trained model."""
        # The model expects batched input, so reshape the single state
        action, _ = self.model.predict(state.reshape(1, -1), deterministic=True)
        self.store_action(action)
        block_size = get_block_size(action, n)
        return block_size

    def store_action(self, action):
        """Call this during evaluation or solving to log the chosen action."""
        if isinstance(action, np.ndarray):
            self.actions_taken.append(action.item())
        else:
            self.actions_taken.append(action)

    def get_actions(self):
        """Returns the list of all stored actions."""
        return self.actions_taken

    def clear_actions(self):
        """Clears the stored actions."""
        self.actions_taken.clear()
