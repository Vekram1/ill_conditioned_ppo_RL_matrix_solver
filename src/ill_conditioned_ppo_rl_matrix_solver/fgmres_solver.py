import numpy as np
import scipy.sparse.linalg as spla


def block_qr_precondition(A, r, block_size):
    """
    Apply block QR preconditioning to residual vector r using blocks from A.
    Returns preconditioned residual vector z.
    """
    n = A.shape[0]
    z = np.zeros_like(r)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        A_block = A[start:end, start:end]
        r_block = r[start:end]

        if A_block.shape[0] == 0 or A_block.shape[1] == 0:
            continue

        try:
            Q, R = np.linalg.qr(A_block)
            y = np.linalg.solve(R, Q.T @ r_block)
            z[start:end] = y
        except np.linalg.LinAlgError:
            z[start:end] = r_block  # fallback

    return z


def arnoldi_iteration(A, M_inv, r0, restart):
    """
    Perform Arnoldi iteration with preconditioning.
    Inputs:
        A: system matrix
        M_inv: preconditioner function that applies M⁻¹
        r0: initial residual
        restart: number of Krylov vectors to generate
    Returns:
        V: orthonormal basis of Krylov subspace
        H: upper Hessenberg matrix
    """
    n = A.shape[0]
    V = np.zeros((n, restart + 1))
    H = np.zeros((restart + 1, restart))

    beta = np.linalg.norm(r0)
    V[:, 0] = r0 / beta

    for j in range(restart):
        z = M_inv(V[:, j])  # apply preconditioner
        w = A @ z
        for i in range(j + 1):
            H[i, j] = np.dot(w, V[:, i])
            w -= H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < 1e-12:
            break  
        V[:, j + 1] = w / H[j + 1, j]

    return V, H, beta


def fgmres_step(A, b, x0, block_size, restart):
    """
    Perform one restarted FGMRES cycle with QR block preconditioning.
    Returns updated solution x_new.
    """
    r0 = b - A @ x0

    def M_inv(r):
        return block_qr_precondition(A, r, block_size)

    V, H, beta = arnoldi_iteration(A, M_inv, r0, restart)

    e1 = np.zeros((restart + 1,))
    e1[0] = beta

    y, *_ = np.linalg.lstsq(H[:restart + 1, :restart], e1[:restart + 1], rcond=None)
    Z = np.column_stack([M_inv(V[:, j]) for j in range(restart)])

    x_new = x0 + Z @ y
    return x_new