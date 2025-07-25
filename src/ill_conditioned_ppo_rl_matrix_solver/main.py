import problem_generator as pg
import gmres_solver

if __name__ == "__main__":
    A = pg.matrix_A() # create and load matrix A
    b = pg.vector_b(A) # create vector b based on matrix A
    x = gmres_solver.gmres_solver(A.matrix, b.vector, maxiter=1000)
