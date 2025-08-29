import problem_generator as pg
import gmres_solver

from stable_baselines3.common.env_checker import check_env
from ppo_trainer import PPOTrainer
from fgmres_env import FGMRESEnv
from fgmres_solver import fgmres_solver
import matplotlib.pyplot as plt
import time
import numpy as np
import cProfile, pstats
from stable_baselines3.common.vec_env import DummyVecEnv
from fgmres_env import get_block_size

TRAINING = 0
TEST = 1

def save_trained_agent(trainer, filename="ppo_fgmres.zip"):
    """
    Save the trained PPO agent to a file.
    """
    trainer.save(filename)
    print(f"Trained agent saved to {filename}")


if __name__ == "__main__":
    if TRAINING:
        TRAIN_MATRICES = [
            ("Newman", "polbooks"),
            ("Grund", "meg1"),
            ("Newman", "polblogs"),
            ("Grund", "b_dyn"),
            ("Marini", "eurqsa"),
            ("Hollinger", "g7jac010sc"),
            ("Grund", "poli_large")
        ]

        TOTAL_TIMESTEPS = 10000
        TIMESTEPS_PER_MATRIX = TOTAL_TIMESTEPS // len(TRAIN_MATRICES)

        # 1. Initialize the PPOTrainer once
        trainer = PPOTrainer()

        # 2. Sequential training loop
        print("--- Starting Sequential PPO Training ---")
        for group, name in TRAIN_MATRICES:
            print(f"\nTraining on matrix: {group}/{name}")

            # Load the current matrix
            matrix_A = pg.matrix_A()
            matrix_A.load_matrix_A(group=group, name=name)
            vector_b = pg.vector_b(matrix_A)

            # Create a single environment for this matrix
            env = FGMRESEnv(matrix_A.matrix, vector_b.vector)
            check_env(env)

            # Train the agent on this specific environment
            trainer.train(env, total_timesteps=TIMESTEPS_PER_MATRIX)

        print("\n--- Training complete ---")

        # 3. Save the final trained agent
        trainer.save()


        # train_A = pg.matrix_A() # create and load matrix A
        # train_A.load_matrix_A(group="Grund",
        #                       name="poli")  # load matrix A from file
        # train_b = pg.vector_b(train_A) # create vector b based on matrix A
        
        # # 2. Initialize and check environment
        # env = FGMRESEnv(train_A.matrix, train_b.vector)
        # check_env(env)

        # # 3. Train PPO agent on FGMRES environment
        # trainer = PPOTrainer(train_A.matrix, train_b.vector, total_timesteps=10000, n_envs=1)
        # trainer.train()

        # #trainer.evaluate(episodes=1)

        # save_trained_agent(trainer, filename="ppo_fgmres.zip")

    if TEST:
        test_A = pg.matrix_A()
        test_A.load_matrix_A(group="Grund", name="poli3")
        test_b = pg.vector_b(test_A)

        # --- Baseline GMRES ---
        start_time_gmres = time.perf_counter()
        try:
            x, residuals = gmres_solver.gmres_solver(
                test_A.matrix.toarray(), test_b.vector, maxiter=10000
            )
            end_time_gmres = time.perf_counter()
            with open("file.txt", "w") as f:
                f.write(
                    f"GMRES time: {end_time_gmres - start_time_gmres:.2f} seconds\n"
                    f"iterations: {len(residuals)}\n"
                    f"final residual norm: {residuals[-1]:.2e}\n"
                )
        except ValueError as e:
            with open("file.txt", "w") as f:
                f.write(f"GMRES failed: {str(e)}\n")

        # --- PPO FGMRES ---
        tester_ppo = PPOTrainer(n_envs=1)
        tester_ppo.load("ppo_fgmres.zip")

        x = np.zeros(test_A.matrix.shape[0])
        print("Starting FGMRES with PPO agent...")
        start_time_fgmres = time.perf_counter()

        x, res_norms, res_at_iters, actions_at_iters = fgmres_solver(
            test_A.matrix.toarray(),
            test_b.vector,
            x,
            ppo_agent=tester_ppo,
            tol=1e-5,
            max_iters=30000,
            restart=10
        )

        end_time_fgmres = time.perf_counter()
        with open("file.txt", "a") as f:
            f.write(
                f"FGMRES with PPO time: {end_time_fgmres - start_time_fgmres:.2f} seconds\n")
            f.write(f"FGMRES iterations: {len(res_norms)}\n")
            f.write(f"FGMRES final residual norm: {res_norms[-1]:.2e}\n")

        # --- Fixed block size ---
        x = np.zeros(test_A.matrix.shape[0])
        print("Starting FGMRES with fixed block size...")
        fixed_block_size = get_block_size(5, test_A.matrix.shape[0])  # Use a fixed block size
        start_time_fgmres_fixed = time.perf_counter()

        x_fixed, res_norms_fixed, res_at_iters_fixed, actions_at_iters_fixed = fgmres_solver(
            test_A.matrix.toarray(),
            test_b.vector,
            x,
            ppo_agent=None,
            tol=1e-5,
            max_iters=30000,
            restart=10,
            block_size=fixed_block_size
        )

        end_time_fgmres_fixed = time.perf_counter()
        with open("file.txt", "a") as f:
            f.write(
                f"FGMRES fixed block time: {end_time_fgmres_fixed - start_time_fgmres_fixed:.2f} seconds\n")
            f.write(f"FGMRES fixed block iterations: {len(res_norms_fixed)}\n")
            f.write(f"FGMRES fixed block final residual norm: {res_norms_fixed[-1]:.2e}\n")

        # --- Plot results ---
        plt.figure(figsize=(10, 5))
        plt.semilogy(res_norms, label='FGMRES with PPO', marker='o')
        plt.semilogy(res_norms_fixed,
                    label=f'FGMRES fixed block {fixed_block_size}', marker='x')
        plt.xlabel('Iteration')
        plt.ylabel('Residual Norm (log scale)')
        plt.title('FGMRES Convergence Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig("fgmres_convergence_comparison.png")
        plt.show()