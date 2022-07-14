"""Module containing ExceptionalPoint class"""
import numpy as np
from scipy.linalg import lu


class ExceptionalPoint():
    """Class defining an exceptional point for the ParallelDot system."""

    def __init__(self, sys, method):
        """Initiates Exceptional point

        Parameters:
        sys -- ParallelDots object at EP
        method -- subspace/full space/inverse, method of finding generalized
                  eigenvectors

        Throws:
        ValueError -- If system not at EP, within some tolerance.
        """
        self.eigvals = sys.eigvals
        # only one exc point atm
        try:
            self.indices = sys.check_if_exc_point()[0]
        except IndexError as error:
            raise ValueError('System not at exceptional point') from error

        self.L, self.R = self.calc_gen_eigvecs(sys.l_eigvecs, sys.r_eigvecs,
                                               method, sys.kern)
        self.consts = None

    def calc_gen_eigvecs(self, l_eigvecs, r_eigvecs, method, kern):
        """Calculates the generalized eigenvectors.

        Parameters:
        l_eigvecs -- left eigenvectors
        r_eigvecs -- right eigenvectors
        method -- subspace/full space/inverse, method of finding generalized
                  eigenvectors
        kern -- the Liouvillian

        Returns:
        L, R -- matrices of left and right generalizd eigenvectors constructed
                biorthogonally. notice that the rows of L.conj().T are the
                proper left generalized eigenvectors.

                L = (l_1, l_2, ... l', l_bar, ... l_q), where l_i.conj().T is
                                                        the proper eigenvector
                R = (r_1, r_2, ... r_bar, r', ... r_q)

                'bar' meaning proper eigenvector and prime meaning chain vector
        """

        ind1 = self.indices[0]
        ind2 = self.indices[1]
        R = r_eigvecs.astype('complex')
        L = l_eigvecs.astype('complex')

        rho_bar = R[:, ind1]
        sigma_prime = L[:, ind2]
        eig_id = self.eigvals[ind1]*np.identity(len(self.eigvals))
        rho_prime = np.linalg.pinv(kern - eig_id) @ rho_bar
        sigma_bar = np.linalg.pinv((kern - eig_id).conj().T) @ sigma_prime
        R[:, ind2] = rho_prime
        L[:, ind1] = sigma_bar

        if method == 'subspace':
            L_subspace = np.array([sigma_bar, sigma_prime]).T
            R_subspace = np.array([rho_bar, rho_prime]).T
            M = L_subspace.conj().T @ R_subspace
            Mp, Ml, Mu = lu(M)
            Lprime = np.linalg.solve(Ml, L_subspace.conj().T).conj().T
            # does this work?
            Rprime = np.linalg.solve(Mu.conj().T, R_subspace.conj().T).conj().T
            R[:, ind1] = Rprime[:, 0]
            R[:, ind2] = Rprime[:, 1]
            L[:, ind1] = Lprime[:, 0]
            L[:, ind2] = Lprime[:, 1]
        elif method == 'full space':
            M = L.conj().T @ R
            Mp, Ml, Mu = lu(M)
            Lprime = np.linalg.solve(Ml, L.conj().T).conj().T
            Rprime = np.linalg.solve(Mu.conj().T, R.conj().T).conj().T
            R = Rprime
            L = Lprime
        elif method == 'inverse':
            L = np.linalg.inv(R).conj().T
        else:
            raise Exception('Not a valid method')

        for i in range(len(self.eigvals)):
            # normalize L s.t L.conj()T@R = I
            L[:, i] /= np.vdot(L[:, i], R[:, i]).conj()

        return L, R

    def calc_constants(self, rho_0):
        """Calculates constants used for density matrix evolution.

        Parameters:
        rho_0 -- initial density matrix

        Returns:
        constants -- vector of the c_i constants
        """
        size = len(rho_0)
        constants = np.array([np.vdot(self.L[:, i], rho_0)
                              for i in range(size)])

        return constants
