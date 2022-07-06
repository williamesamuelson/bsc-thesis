import numpy as np
from scipy.linalg import lu
# from parallel_dots import ParallelDots


class ExceptionalPoint():

    def __init__(self, system):
        self.eigvals = system.eigvals
        self.r_eigvecs = system.r_eigvecs
        self.l_eigvecs = system.l_eigvecs
        # only one exc point atm
        self.indices = system.check_if_exc_point()[0]
        self.kern = system.kern

    def calc_gen_eigvecs(self):
        size = len(self.eigvals)
        R = self.r_eigvecs.copy()
        L = self.l_eigvecs.copy()

        rho5 = R[:, self.indices[0]]
        sigma_prime = L[:, self.indices[0]]
        eig_id = self.eigvals[self.indices[0]]*np.identity(size)
        rho_prime = np.linalg.pinv(self.kern - eig_id) @ rho5
        sigma5 = np.linalg.pinv((self.kern - eig_id).conj().T) @ sigma_prime
        R[:, self.indices[1]] = rho_prime
        L[:, self.indices[1]] = sigma5
        # nånting
        L_subspace = np.array()
        R_subspace = R[:, self.indices[0]:self.indices[1]+1]
        M = L_subspace @ R_subspace
        Mp, Ml, Mu = lu(M)
        Lprime = np.linalg.solve(Ml, L_subspace)
        Rprime = np.linalg.solve(Mu.conj().T, R_subspace.conj().T).conj().T
        print(Lprime @ Rprime)


