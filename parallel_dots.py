import numpy as np
from qmeq.builder.builder import Builder


class ParallelDots(Builder):

    def __init__(self, gamma, delta_eps, delta_t,
                 d_vec, kerntype, parameters='stephanie', v_bias=None):
        self.gamma = gamma
        self.vgate = 0
        self.d_vec = d_vec

        if parameters == 'stephanie':
            vbias = 30*gamma
            temp = 10*gamma
            U = 250*gamma
        elif parameters == 'paper':
            vbias = 1e4*gamma
            temp = 862*gamma
            U = 2e4*gamma

        if vbias is not None:
            vbias = v_bias

        dband = 12*U

        # number of single particle states
        nsingle = 2

        # number of leads
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

    def change_delta_eps(self, delta_eps):
        new_hsingle = self._create_hsingle(delta_eps)
        self.change(hsingle=new_hsingle)

    def change_delta_t(self, delta_t):
        new_tleads = self._create_tleads(delta_t)
        self.change(tleads=new_tleads)

    def _create_hsingle(self, delta_eps):
        # single particle hamiltonian, 0: upper, 1: lower
        hsingle = {(0, 0): -self.vgate + delta_eps,
                   (1, 1): -self.vgate - delta_eps}
        return hsingle

    def _create_tleads(self, delta_t):
        delta_gammas = self.d_vec*delta_t
        t0s = np.sqrt((self.gamma + delta_gammas)/(2*np.pi))
        # single particle tunneling amplitudes,
        # (Lead L/R 0/1, Dot upper/lower 0/1)
        tleads = {(0, 0): t0s[0],  # L <- upper
                  (1, 0): t0s[1],  # R <- upper
                  (0, 1): t0s[2],  # L <- lower
                  (1, 1): t0s[3]}  # R <- lower
        return tleads
