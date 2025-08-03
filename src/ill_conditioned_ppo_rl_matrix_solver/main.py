import problem_generator as pg
import gmres_solver

from stable_baselines3.common.env_checker import check_env
from ppo_trainer import PPOTrainer
from fgmres_env import FGMRESEnv
from fgmres_solver import fgmres_solver

import time

TRAINING = 1
TEST = 1

def save_trained_agent(trainer, filename="ppo_fgmres.zip"):
    """
    Save the trained PPO agent to a file.
    """
    trainer.save(filename)
    print(f"Trained agent saved to {filename}")


if __name__ == "__main__":
    if TRAINING:
        train_A = pg.matrix_A() # create and load matrix A
        train_A.load_matrix_A(group="Grund",
                              name="Poli")  # load matrix A from file
        train_b = pg.vector_b(train_A) # create vector b based on matrix A
        
        # 2. Initialize and check environment
        env = FGMRESEnv(train_A.matrix, train_b.vector)
        check_env(env)

        # 3. Train PPO agent on FGMRES environment
        trainer = PPOTrainer(train_A.matrix, train_b.vector, total_timesteps=10000, n_envs=5)
        trainer.train()

        #trainer.evaluate(episodes=1)

        save_trained_agent(trainer, filename="ppo_fgmres.zip")


    if TEST:
        test_A = pg.matrix_A() # create and load matrix A
        test_A.load_matrix_A(group="Grund", name="Poli")  # load matrix A from file
        test_b = pg.vector_b(test_A) # create vector b based on matrix A

        start_time_gmres = time.perf_counter()
        x, residuals = gmres_solver.gmres_solver(test_A.matrix, test_b.vector, maxiter=1000)
        end_time_gmres = time.perf_counter()
        with open("file.txt", "w") as f:
            f.write(
                f"GMRES time: {end_time_gmres - start_time_gmres:.2f} seconds\niterations: {len(residuals)}\nfinal residual norm: {residuals[-1]:.2e}\n")



        # 4. Use trained agent inside fgmres_solver
        #trained_agent = trainer.model
        #x, res_norms = fgmres_solver(A.matrix, b.vector, ppo_agent=trained_agent)
        tester_ppo = PPOTrainer(test_A.matrix, test_b.vector, total_timesteps=0, n_envs=1)
        tester_ppo.load("ppo_fgmres.zip")


        print("Starting FGMRES with PPO agent...")
        start_time_fgmres = time.perf_counter()
        x, res_norms, res_at_iters, actions_at_iters = fgmres_solver(test_A.matrix.toarray(), test_b.vector, ppo_agent=tester_ppo,  tol=1e-6, max_iters=30, restart=10)
        end_time_fgmres = time.perf_counter()
        # print("Actions taken during solve:", trainer.get_actions())

        # 5. Output final residual norm
        # print(f"Final residual norm: {res_norms[-1]:.2e}")
        # print(res_at_iters)
        # print(actions_at_iters)

        with open("file.txt", "a") as f:
            f.write(f"FGMRES with PPO time: {end_time_fgmres - start_time_fgmres:.2f} seconds\n")
            f.write(f"FGMRES iterations: {len(res_norms)}\n")
            f.write(f"FGMRES final residual norm: {res_norms[-1]:.2e}\n")