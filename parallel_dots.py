"""Module containing ParallelDots class extending Builder"""

import numpy as np
from scipy.linalg import eig as sc_eig
from qmeq.builder.builder import Builder
import myLindblad


class ParallelDots(Builder):
    """ Subclass of Builder representing a parallel double quantum dot system.
    """

    def __init__(self, gamma, delta_eps, delta_t,
                 d_vec, rho_0, kerntype, parameters='stephanie', v_bias=None):
        """Initiates the parallel dot system by creating a Builder object.

        Arguments:
        gamma -- tunneling rate
        delta_eps -- offset energy of the quantum dots
        delta_t -- offset tunneling rates of the quantum dots
        d_vec -- which tunneling rates which are affected [L <- upper,
                 R <- upper, L <- lower, R <- lower] (4,)
        rho_0 -- initial condition of density matrix (6,)
        kerntype -- the type of master equation approach used
        parameters -- standard parameters, 'stephanie' or 'paper'
        v_bias -- can change bias voltage from standard parameters
       """

        self.gamma = gamma
        self.vgate = 0
        self.d_vec = d_vec
        self.eigvals = None
        self.r_eigvecs = None
        self.l_eigvecs = None
        self.rho_0 = rho_0

        if parameters == 'stephanie':
            vbias = 30*gamma
            temp = 10*gamma
            U = 250*gamma
        elif parameters == 'paper':
            vbias = 1e4*gamma
            temp = 862*gamma
            U = 2e4*gamma

        if v_bias is not None:
            vbias = v_bias

        # dband = 12*U
        dband = 1e10
        nsingle = 2
        nleads = 2
        hsingle = self._create_hsingle(delta_eps)
        tleads = self._create_tleads(delta_t)

        # Coulomb matrix elements
        coulomb = {(0, 1, 1, 0): U}

        # chemical potentials and temperatures of the leads
        #           L         R
        mulst = {0: vbias/2, 1: -vbias/2}
        tlst = {0: temp, 1: temp}

        # python kerntype + make_kern_copy fixes problem with kernel
        super().__init__(nsingle, hsingle, coulomb, nleads, tleads,
                         mulst, tlst, dband,
                         kerntype=kerntype, itype=1)
        self.make_kern_copy = True

    def solve(self, qdq=True, rotateq=True, masterq=True, currentq=True,
              sol_eig=True, sort=True, lamb_shift=True, *args, **kwargs):
        """Solves the master equation and sets the kernel matrix.

        Extends solve() from Builder (approach) by also creating eigenvectors,
        eigenvalues and includes Lamb shift from myLindblad.py, all optional.

        Parameters:
        qdq -- True to diagonalize
        rotateq -- ?
        masterq -- True to solve master eq
        currentq -- True to calculate current
        sol_eig -- True to calculate eigenvectors
        sort -- True to sort eigenvectors
        lamb_shift -- True to include Lamb shift
        """

        self.appr.solve(qdq=qdq, rotateq=rotateq, masterq=masterq,
                        currentq=currentq, *args, **kwargs)

        if lamb_shift:
            self.kern = myLindblad.calc_Lindblad_kernel(self)

        if sol_eig:  # dont we want to do this every time?
            self.eigvals, self.l_eigvecs, self.r_eigvecs = sc_eig(self.kern,
                                                                  left=True,
                                                                  right=True)
            if sort:
                indices = np.argsort(self.eigvals)
                self.eigvals = self.eigvals[indices]
                self.l_eigvecs = self.l_eigvecs[:, indices]
                self.r_eigvecs = self.r_eigvecs[:, indices]

        for i in range(len(self.eigvals)):
            scalar_prod = np.vdot(self.l_eigvecs[:, i], self.r_eigvecs[:, i])
            if scalar_prod != 1:
                # normalize them somehow? Seems like they are orthogonal,
                # but l_i.H dot r_i does not give 1?
                pass

    def change_delta_eps(self, delta_eps):
        """Changes delta_eps of the system

        Parameters:
        delta_eps -- the new value of delta_epsilon
        """
        new_hsingle = self._create_hsingle(delta_eps)
        self.change(hsingle=new_hsingle)

    def change_delta_t(self, delta_t):
        """Changes delta_t of the system

        Parameters:
        delta_t -- the new value of delta_t
        """
        new_tleads = self._create_tleads(delta_t)
        self.change(tleads=new_tleads)

    def _create_hsingle(self, delta_eps):
        """Creates single particle Hamiltonian.

        Parameters:
        delta_eps -- delta_epsilon of the system

        Returns:
        hsingle -- hsingle dictionary which goes into creating system
        """

        # single particle hamiltonian, 0: upper, 1: lower
        hsingle = {(0, 0): -self.vgate - delta_eps,
                   (1, 1): -self.vgate + delta_eps}
        return hsingle

    def _create_tleads(self, delta_t):
        """Creates the tunneling part of the Hamiltonian.

        Parameters:
        delta_t -- tunneling offset

        Returns:
        tleads -- tleads dictionary which goes into creating system.
        """

        delta_gammas = self.d_vec*delta_t
        t0s = np.sqrt((self.gamma + delta_gammas)/(2*np.pi))
        # single particle tunneling amplitudes,
        # (Lead L/R 0/1, Dot upper/lower 0/1)
        tleads = {(0, 0): t0s[0],  # L <- upper
                  (1, 0): t0s[1],  # R <- upper
                  (0, 1): t0s[2],  # L <- lower
                  (1, 1): t0s[3]}  # R <- lower
        return tleads

    def check_if_exc_point(self, eigenval_tol=0.1, eigenvec_tol=0.1):
        """ Checks if system is at an exceptional point.

        Parameters:
        eigenval_tol -- tolerance for eigenvalue merge
        eigenvec_tol -- tolerance for eigenvector merge

        Returns:
        indices -- pairs of indices which describe which eigenvectors
                   correspond to exceptional point. The indices correspond to
                   the ones in self.eigenvalues. Empty if not at an
                   exceptional point.
        """
        eigenvals, eigenvecs = self.eigvals, self.r_eigvecs

        # get indices of pairs of potential eigenvectors corr. to exc. point
        potential_indices = []
        # Loop and compare eigenval i with all eigenvals to the right of i.
        # If smaller seperation than eigenval_tol, add the pair of indices to
        # potential_indices.
        for i in range(len(eigenvals)-1):
            curr_seps = np.abs(eigenvals[i] - eigenvals[i+1::])
            curr_min_sep_ind = np.argmin(curr_seps)
            if curr_seps[curr_min_sep_ind] < eigenval_tol:
                potential_indices.append((i, i+curr_min_sep_ind+1))

        # Make copy of potential_indices and loop over it ([:]). This way one
        # can remove elements in the original list when iterating over it
        for i, j in potential_indices[:]:
            distance = np.linalg.norm(eigenvecs[:, i]-eigenvecs[:, j])
            if distance > eigenvec_tol:
                # remove from potential_indices
                potential_indices.remove((i, j))

        # returns empty list if no exceptional point
        return potential_indices

    def dens_matrix_evo(self, t):
        """Calculates the density matrix at time t.

        Parameters:
        t -- time

        Returns:
        dens_evol -- the vectorized density matrix at time t in the format
                     [rho_aa, rho_bb, rho_cc, rho_dd, re(rho_bc), im(rho_bc)]

        """
        # something weird with the left and right eigenvectors I think.
        # not orthonormal
        rho_0 = self.rho_0
        eigvals = self.eigvals
        l_eigvecs, r_eigvecs = self.l_eigvecs, self.r_eigvecs
        size = len(self.eigvals)

        # inner products of left eigenvectors and rho_0 to get all c_i's
        constants = np.array([np.vdot(l_eigvecs[:, i], rho_0) for i in
                             range(size)])
        # multiply with all exp(lambda_i*t)
        constants_exp = constants * np.exp(eigvals*t)

        # multiply column i in r_eigvecs with c_i*exp(lambda_i*t)
        summation_matrix = constants_exp * r_eigvecs

        # do the summation over all column vectors
        dens_evol = np.sum(summation_matrix, axis=1)
        return dens_evol
