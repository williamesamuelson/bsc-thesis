import numpy as np
from scipy.linalg import lu
# from parallel_dots import ParallelDots


class ExceptionalPoint():

    def __init__(self, sys, subspace=False):
        self.eigvals = sys.eigvals
        # only one exc point atm
        try:
            self.indices = sys.check_if_exc_point()[0]
        except:
            raise Exception('System not at exceptional point')

        self.L, self.R = self.calc_gen_eigvecs(sys.l_eigvecs, sys.r_eigvecs,
                                               subspace, sys.kern)
        self.consts = None

    def calc_gen_eigvecs(self, l_eigvecs, r_eigvecs, subspace, kern):
        size = len(self.eigvals)
        ind1 = self.indices[0]
        ind2 = self.indices[1]
        R = r_eigvecs.astype('complex')
        L = l_eigvecs.astype('complex')

        rho5 = R[:, ind1]
        sigma_prime = L[:, ind2]
        eig_id = self.eigvals[ind1]*np.identity(size)
        rho_prime = np.linalg.pinv(kern - eig_id) @ rho5
        sigma5 = np.linalg.pinv((kern - eig_id).conj().T) @ sigma_prime
        R[:, ind2] = rho_prime
        L[:, ind1] = sigma5

        # L = L.conj().T

        if subspace:

            L_subspace = np.array([sigma5, sigma_prime]).T
            R_subspace = np.array([rho5, rho_prime]).T
            M = L_subspace.conj().T @ R_subspace
            Mp, Ml, Mu = lu(M)
            Lprime = np.linalg.solve(Ml, L_subspace.conj().T).conj().T
            # does this work?
            Rprime = np.linalg.solve(Mu.conj().T, R_subspace.conj().T).conj().T
            R[:, ind1] = Rprime[:, 0]
            R[:, ind2] = Rprime[:, 1]
            L[:, ind1] = Lprime[:, 0]
            L[:, ind2] = Lprime[:, 1]
        else:
            M = L.conj().T @ R
            Mp, Ml, Mu = lu(M)
            Lprime = np.linalg.solve(Ml, L.conj().T).conj().T
            Rprime = np.linalg.solve(Mu.conj().T, R.conj().T).conj().T
            R = Rprime
            L = Lprime

        for i in range(size):
            L[:, i] /= np.vdot(L[:, i], R[:, i]).conj()

        return L, R

    def calc_constants(self, rho_0):
        size = len(rho_0)
        constants = np.array([np.vdot(self.L[:, i], rho_0)
                              for i in range(size)])

        return constants
        
