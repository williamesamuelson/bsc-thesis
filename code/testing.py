import numpy as np
from scipy.linalg import eig as sc_eig
from scipy.linalg import lu

A = np.array([[0, 4, 2], [-3, 8, 3], [4, -8, -2]])

w, l, r = sc_eig(A, left=True)


rx1 = r[:, 0]
lx1 = l[:, 0]
ry1 = r[:, 2]
ly1 = l[:, 2]
rx2 = np.linalg.pinv(A-2*np.identity(len(w))) @ rx1
lx2 = np.linalg.pinv((A-2*np.identity(len(w))).conj().T) @ lx1
R = np.array([ry1, rx1, rx2]).T
# Reverse x1 and x2?? look paper
L = np.array([ly1, lx2, lx1]).conj()
M = L@R
Mp, Ml, Mu = lu(M)
Lprime = np.linalg.solve(Ml, L)
Rprime = np.linalg.solve(Mu.conj().T, R.conj().T).conj().T
print(Lprime@Rprime)
