from time import time
import numpy as np
from typing import List
from cvxopt import matrix, spmatrix, spdiag, sparse
from cvxopt import solvers
import matplotlib.pyplot as plt

N = 500

Aeq_ = matrix(1., (1, 2))  # equality constraint x1 + x2 = 1
beq_ = matrix(1.)

# box-bounds
lb = np.array([.2, .2])
ub = np.array([.8, .8])

x = np.random.uniform(size=(N, 2))


def vstack(m: List[matrix]) -> matrix:
    return matrix([[mi] for mi in m])

def repeated_blkdiag(m: matrix, N:int) -> matrix:
    m_zeros = spmatrix([], [], [], m.size)
    return sparse([[m_zeros] * i + [m] + [m_zeros] * (N - i - 1) for i in range(N)])


# inequalitites
Aeq = repeated_blkdiag(Aeq_, N)
beq = matrix([beq_]*N)


# box-bounds
G_bounds_ = sparse([spmatrix(1, range(2), range(2)), spmatrix(-1, range(2), range(2))])
h_bounds_ = matrix(np.concatenate((ub.reshape(-1), lb.reshape(-1))))
G = repeated_blkdiag(G_bounds_, N)
h = matrix([h_bounds_]*N)

# A = repeated_blkdiag(A_, N)
# b = matrix([b_]*N)

q = matrix(-1.*x.reshape(-1))

P = spmatrix(1.0, range(2*N), range(2*N))  # the unit-matrix

t0 = time()
sol = solvers.qp(P, q, A=Aeq, b=beq, G=G, h=h)
print(f"problem solved in {time()-t0} s")
# sol = coneqp(P, q, G=Aeq, h=beq, A=A, b=b, )

xs = np.array(sol["x"]).transpose().reshape((-1, 2))


plt.figure()
for i in range(N):
    plt.plot((x[i, 0], xs[i, 0]), (x[i, 1], xs[i, 1]), '-', c='grey', lw=.3)
plt.scatter(xs[:, 0], xs[:, 1], color="red", marker="x")
#plt.scatter(x[:, 0], x[:, 1], color="green", marker="o")
plt.grid()
plt.xlim((0., 1.))
plt.ylim((0., 1.))
plt.show()

xs.sum(axis=1).sum()