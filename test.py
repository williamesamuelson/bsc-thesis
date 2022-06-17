import numpy as np
import scipy
import matplotlib.pyplot as plt
import qmeq
from parallel_dots import ParallelDots


def stab_calc(system, vlst, vglst, delta_eps, dV=0.001):
    vpnt, vgpnt = vlst.shape[0], vglst.shape[0]
    stab = np.zeros((vpnt, vgpnt))
    stab_cond = stab.copy()

    for j1 in range(vgpnt):
        system.change(hsingle={(0, 0): -vglst[j1] + delta_eps,
                               (1, 1): -vglst[j1] - delta_eps})
        system.solve(masterq=False)
        for j2 in range(vpnt):
            system.change(mulst={0: vlst[j2]/2, 1: -vlst[j2]/2})
            system.solve(qdq=False)
            stab[j1, j2] = system.current[0]

            system.add(mulst={0: dV/2, 1: -dV/2})
            system.solve(qdq=False)
            stab_cond[j1, j2] = (system.current[0] - stab[j1, j2])/dV
    return stab, stab_cond


def stab_plot(stab, stab_cond, vlst, vglst, U, gam):
    (xmin, xmax, ymin, ymax) = np.array([vglst[0], vglst[-1],
                                         vlst[0], vlst[-1]])/gam
    plt.figure(figsize=(12, 4.2))
    #
    p1 = plt.subplot(1, 2, 1)
    p1.set_xlabel(r'$V_{g}/\Gamma$', fontsize=20)
    p1.set_ylabel(r'$V/\Gamma$', fontsize=20)
    p1.ticklabel_format(style='scientific', scilimits=(-2, 2))
    p1_im = plt.imshow(stab.T/gam, extent=[xmin, xmax, ymin, ymax],
                       aspect='auto', origin='lower', cmap='bwr')
    cbar1 = plt.colorbar(p1_im)
    cbar1.set_label(r'Current [$\Gamma$]', fontsize=20)
    cbar1.formatter.set_powerlimits((-2, 2))
    #
    p2 = plt.subplot(1, 2, 2)
    p2.set_xlabel(r'$V_{g}/\Gamma$', fontsize=20)
    p2.set_ylabel(r'$V/\Gamma$', fontsize=20)
    p2.ticklabel_format(style='scientific', scilimits=(-2, 2))
    p2_im = plt.imshow(stab_cond.T, extent=[xmin, xmax, ymin, ymax],
                       aspect='auto', origin='lower', cmap='bwr')
    cbar2 = plt.colorbar(p2_im)
    cbar2.set_label(r'Conductance $\mathrm{d}I/\mathrm{d}V$', fontsize=20)
    cbar2.formatter.set_powerlimits((-2, 2))
    plt.tight_layout()
    plt.show()


def quick_plot(system, delta_eps, gamma, parameters, U=None):
    if parameters == 'stephanie':
        U = 250*gamma
    elif parameters == 'paper':
        U = 2e4*gamma
    dV = 1e-3*gamma
    vpnt, vgpnt = 201, 201
    vlst = np.linspace(-2*U, 2*U, vpnt)
    vglst = np.linspace(-1.25*U, 2.25*U, vgpnt)
    stab, stab_cond = stab_calc(system, vlst, vglst, delta_eps, dV)
    stab_plot(stab, stab_cond, vlst, vglst, U, gamma)


def create_double_dot_system(gamma, parameters, delta_eps,
                             delta_t, d_vec, vbias=None, temp=None, U=None):
    vgate = 0
    if parameters == 'stephanie':
        vbias = 30*gamma
        temp = 10*gamma
        U = 250*gamma
    elif parameters == 'paper':
        vbias = 1e4*gamma
        temp = 862*gamma
        U = 2e4*gamma

    dband = 12*U
    delta_gammas = d_vec*delta_t
    t0s = np.sqrt((gamma + delta_gammas)/(2*np.pi))

    # number of single particle states
    nsingle = 2

    # number of leads
    nleads = 2

    # single particle hamiltonian, 0: upper, 1: lower
    hsingle = {(0, 0): -vgate + delta_eps,
               (1, 1): -vgate - delta_eps}

    # Coulomb matrix elements
    coulomb = {(0, 1, 1, 0): U}

    # single particle tunneling amplitudes, (Lead L/R 0/1, Dot upper/lower 0/1)
    tleads = {(0, 0): t0s[0],  # L <- upper
              (1, 0): t0s[1],  # R <- upper
              (0, 1): t0s[2],  # L <- lower
              (1, 1): t0s[3]}  # R <- lower

    # chemical potentials and temperatures of the leads
    #           L         R
    mulst = {0: vbias/2, 1: -vbias/2}
    tlst = {0: temp, 1: temp}

    # python kerntype + make_kern_copy fixes problem with kernel
    system = qmeq.Builder(nsingle, hsingle, coulomb, nleads, tleads, mulst,
                          tlst, dband, kerntype='py1vN', itype=1)
    system.make_kern_copy = True

    return system


def plot_spectrum(eigvals):
    im_parts = [eigval.imag for eigval in eigvals]
    re_parts = [eigval.real for eigval in eigvals]

    fig, ax = plt.subplots()
    spec_fmt = {'color': 'blue', 'ls': 'none', 'marker': '.', 'markersize': 15}
    ax.plot(re_parts, im_parts, **spec_fmt)
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.show(block=False)


def plot_tuning(delta_epsilons, system):
    system_size = system.kern.shape[0]
    eigs = np.zeros((len(delta_epsilons), system_size), dtype=complex)

    for i, delta_e in enumerate(delta_epsilons):
        system.change_delta_eps(delta_e)
        system.solve()
        current_eigvals = np.linalg.eigvals(system.kern)
        # sorting messes it up if they cross
        eigs[i, :] = np.sort(current_eigvals)

    fig, (ax_real, ax_imag, ax_spec) = plt.subplots(1, 3, figsize=(15, 8))
    for eig_index in range(system_size): 
        ax_real.plot(delta_epsilons, np.real(eigs[:, eig_index]))
        ax_imag.plot(delta_epsilons, np.imag(eigs[:, eig_index]))

        im_parts = [eigval.imag for eigval in eigs[:, eig_index]]
        re_parts = [eigval.real for eigval in eigs[:, eig_index]]
        ax_spec.scatter(re_parts, im_parts, s=np.linspace(2, 8, len(delta_epsilons)))
    plt.show(block=False)
    ax_real.legend(np.arange(system_size))
    ax_imag.legend(np.arange(system_size))




if __name__ == '__main__':
    gamma = 1
    delta_eps = gamma*1e-3
    delta_t = gamma*1e-3
    v_bias = 400*gamma

    #           upper->L, upper->R, lower->L, lower->R
    d_vec = np.array([-1, 1, 1, -1])
    parameters = 'stephanie'
    parallel_dots_1vN = ParallelDots(gamma, delta_eps, delta_t, d_vec,
                                     'py1vN', parameters, v_bias)
    parallel_dots_lind = ParallelDots(gamma, delta_eps, delta_t, d_vec,
                                      'pyLindblad', parameters, v_bias)
    parallel_dots_1vN.solve()
    parallel_dots_lind.solve()
    delta_epsilons = np.linspace(0.2, 0.4, 100)
    plot_tuning(delta_epsilons, parallel_dots_1vN)
    plot_tuning(delta_epsilons, parallel_dots_lind)
    
