import numpy as np
from scipy.linalg import qr, solve_triangular
from numpy.linalg import norm


def apply_block_qr_preconditioner(A, v, block_sizes):
    n = A.shape[0]
    x = np.zeros_like(v)
    start = 0

    for size in block_sizes:
        end = min(start + size, n)
        #print(f"[Block QR] Processing block size: {size}, start: {start}, {end}")

        A_block = A[start:end, start:end]
        v_block = v[start:end]

        # Check A_block validity
        if A_block.ndim != 2 or A_block.shape[0] == 0 or A_block.shape[1] == 0:
            print(f"[Warning] Skipping invalid block: shape={A_block.shape}")
            start = end
            continue

        # Ensure it's square
        if A_block.shape[0] != A_block.shape[1]:
            print(f"[Warning] Non-square A_block: shape={A_block.shape}")
            start = end
            continue

        # Apply QR
        try:
            Q, R = qr(A_block)
        except ValueError as e:
            print(f"[Error] QR decomposition failed for block {start}:{end} {A_block}: {e}")
        
        y_block = solve_triangular(R, Q.T @ v_block)
        x[start:end] = y_block

        start = end

    return x



def arnoldi_iteration(A, M, b, restart):
    n = A.shape[0]
    V = np.zeros((n, restart + 1))
    H = np.zeros((restart + 1, restart))

    beta = norm(b)
    V[:, 0] = b / beta

    for j in range(restart):
        w = M(V[:, j])  # Apply preconditioner to V[:, j]
        w = A @ w

        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i], w)
            w -= H[i, j] * V[:, i]

        H[j + 1, j] = norm(w)
        if H[j + 1, j] < 1e-14:
            return V, H, j + 1
        V[:, j + 1] = w / H[j + 1, j]

    return V, H, restart


def fgmres_solver(A, b, ppo_agent=None, tol=1e-6, max_iters=20, restart=10, block_size=-1):
    """
    Flexible GMRES solver with optional PPO-guided block preconditioning.
    
    Args:
        A: Square numpy array or matrix (n x n)
        b: Right-hand side vector (n,)
        ppo_agent: PPO agent with select_action() and store_transition() methods
        tol: Convergence tolerance on the residual norm
        max_iters: Maximum FGMRES outer iterations
        restart: Restart parameter for GMRES inner iterations
    
    Returns:
        x: Approximate solution to Ax = b
        res_norms: List of residual norms at each outer iteration
    """
    n = A.shape[0]
    x = np.zeros(n)
    r = b - A @ x
    res_norms = [norm(r)]

    residuals_at_iter = []
    actions_at_iter = []

    for iteration in range(max_iters):
        curr_residual_norm = norm(r)
        if curr_residual_norm < tol:
            break

        state = np.array([curr_residual_norm], dtype=np.float32)

        # If block size has been passed in do nothing, if not and an agent have the agent 
        # select the block size
        if ppo_agent is not None:
            block_size = ppo_agent.select_action(state, n)
            print(f"[FGMRES] Iter {iteration + 1} | Selected block size: {block_size}")

        # Tile block sizes across the matrix diagonal
        block_sizes = []
        remaining = n
        while remaining > 0:
            if block_size <= remaining:
                block_sizes.append(block_size)
                remaining -= block_size
            else:
                block_sizes.append(remaining)
                remaining = 0

        # Define block-preconditioned linear operator
        def preconditioner(v):
            return apply_block_qr_preconditioner(A, v, block_sizes)

        # Arnoldi process with restart
        V, H, k = arnoldi_iteration(A, preconditioner, r, restart)

        # Solve least-squares problem: min_y || beta*e1 - H*y ||
        e1 = np.zeros(k + 1)
        e1[0] = norm(r)
        y, _, _, _ = np.linalg.lstsq(H[:k+1, :k], e1, rcond=None)
        dx = V[:, :k] @ y

        # Update solution and residual
        x += dx
        r = b - A @ x
        res_norm = norm(r)
        res_norms.append(res_norm)

        # RL transition storage
        if ppo_agent is not None:
            reward = -res_norm  # Encourage residual minimization
            next_state = r.copy()
            done = res_norm < tol
            residuals_at_iter.append(res_norm)
            actions_at_iter.append(block_size)
            if done:
                # print(f"finished at iteration {iteration + 1} with residual norm {res_norm:.2e}")
                break

    return x, res_norms, residuals_at_iter, actions_at_iter 
