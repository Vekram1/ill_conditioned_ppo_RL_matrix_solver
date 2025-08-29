import numpy as np
from scipy.linalg import qr, solve_triangular
from numpy.linalg import norm, pinv
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse import csr_matrix
import sparseqr
from scipy.sparse.linalg import spsolve_triangular

def apply_block_qr_preconditioner(A, v, block_sizes):
    n = A.shape[0]
    x = np.zeros_like(v)
    start = 0

    # for size in block_sizes:
    #     end = min(start + size, n)

    #     A_block = A[start:end, start:end]
    #     v_block = v[start:end]

    #     # Sparse QR decomposition with column pivoting
    #     Q, R, E, rank = sparseqr.qr(A_block)

    #     # Step 1: Apply Q^T to v_block (ensure v_block is column vector)
    #     rhs = (Q.T @ v_block.reshape(-1, 1)).ravel()

    #     # Step 2: Solve R z = rhs
    #     # (convert R to csc for efficient triangular solve)
    #     R_csc = R.tocsc()
    #     z = spsolve_triangular(R_csc, rhs, lower=False)

    #     # Step 3: Apply column permutation to scatter back
    #     # E can be a permutation *array* instead of a matrix depending on sparseqr version
    #     if hasattr(E, "toarray"):   # if it's a matrix
    #         x_block = (E @ z.reshape(-1, 1)).ravel()
    #     else:  # if it's an index array
    #         x_block = np.zeros_like(z)
    #         x_block[E] = z

    #     x[start:end] = x_block
    #     start = end

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
            Q, R = qr(A_block) # Creates an orthonormal basis Q and upper triangular matrix R
            rhs = Q.T @ v_block
            try:
                # Fast path: triangular solve
                y_block = solve_triangular(R, rhs, check_finite=False)
            except np.linalg.LinAlgError:
                # Fall back to pseudo-inverse if R is singular
                y_block = pinv(R) @ rhs
        except ValueError as e:
            print(f"[Error] QR decomposition failed for block {start}:{end} {A_block}: {e}")
            y_block = np.zeros_like(v_block)
        
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


def fgmres_solver(A , b, x, ppo_agent=None, tol=1e-6, max_iters=20, restart=10, block_size=-1):
    """
    Flexible GMRES solver with optional PPO-guided block preconditioning.
    
    Args:
        A: csr_matrix 
        b: Right-hand side vector (n,)
        x: The initial guess or current solution vector
        ppo_agent: PPO agent with select_action() and store_transition() methods
        tol: Convergence tolerance on the residual norm
        max_iters: Maximum FGMRES outer iterations
        restart: Restart parameter for GMRES inner iterations
    
    Returns:
        x: Approximate solution to Ax = b
        res_norms: List of residual norms at each outer iteration
    """
    n = A.shape[0]
    A_linop = aslinearoperator(A)  # Use a linear operator for efficiency

    # Calculate the initial residual based on the provided x
    r = b - A_linop @ x
    res_norms = [norm(r)]

    initial_residual_norm = norm(r)
    # This is a key change: if the initial norm is zero, the problem is already solved
    if initial_residual_norm < 1e-14:
        return x, res_norms, [], []

    residuals_at_iter = []
    actions_at_iter = []

    for iteration in range(max_iters):
        curr_residual_norm = norm(r)
        if curr_residual_norm < tol:
            break

        if ppo_agent is not None:
            # Calculate all four observation features here
            relative_residual_norm = np.clip(
                curr_residual_norm / initial_residual_norm, a_min=None, a_max=1.0)
            normalized_n = np.log10(n) / np.log10(4008)  # Assumes max n
            normalized_iter = iteration / max_iters

            state = np.array([
                relative_residual_norm,
                np.log10(relative_residual_norm + 1e-10),
                normalized_n,
                normalized_iter
            ], dtype=np.float32)

            block_size = ppo_agent.select_action(state, n)
            # print(f"[FGMRES] Iter {iteration + 1} | Selected block size: {block_size}, {relative_residual_norm:.2e}, {normalized_n:.2f}, {normalized_iter:.2f}")

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
        r = b - A_linop @ x
        res_norm = norm(r)
        res_norms.append(res_norm)
        #print(f"[FGMRES] Iter {iteration + 1} | Residual norm: {res_norm:.2e}")

        # RL transition storage
        done = res_norm < tol
        residuals_at_iter.append(res_norm)
        actions_at_iter.append(block_size)
        if done:
            print(
                f"finished at iteration {iteration + 1} with residual norm {res_norm:.2e}")
            break

    return x, res_norms, residuals_at_iter, actions_at_iter