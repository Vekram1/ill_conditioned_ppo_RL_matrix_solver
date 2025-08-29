import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

def gmres_solver(A, b, maxiter=None):
    residuals = []

    def callback(rk):
        residuals.append(rk)

    x, info = spla.gmres(A, b, restart=10, rtol=1e-5, maxiter=maxiter, callback=callback, callback_type='legacy')
    if info != 0:
        raise ValueError(f"GMRES failed to converge, info={info}")
    final_residual = np.linalg.norm(b - A @ x)
    # print(f"Final residual norm: {final_residual:.2e}")

    # Plotting
    plt.figure(figsize=(8, 4))
    plt.semilogy(residuals, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm (log scale)')
    plt.title('GMRES Convergence')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return x, residuals
