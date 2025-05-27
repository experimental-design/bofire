
from scipy import sparse
import numpy as np

N = 10
d = 2

x = np.random.rand(N, d)

# objective
P = sparse.identity(N*d)
q = -x.reshape(-1)

# constraints
#Ai = np.array([[1, 1]])
Ai = sparse.csr_array(([1., 1.], [0, 0], [0, 1]), shape=(1, d))
bi = np.array([1])


A = sparse.block_diag([Ai for _ in range(N)])
b = np.vstack([bi for _ in range(N)])

# Solve the quadratic program using cvxpy
import cvxpy as cp
x_var = cp.Variable((N*d, 1))

objective = cp.Minimize(0.5 * cp.quad_form(x_var, P) + q.T @ x_var)

constraints = [A @ x_var == b]

problem = cp.Problem(objective, constraints)

problem.solve(solver=cp.ECOS, verbose=True)

print("Optimal value:", problem.value)

print("Optimal solution:", x_var.value)