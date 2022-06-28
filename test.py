import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from parallel_dots import ParallelDots


def stab_calc(system, vlst, vglst, delta_eps, dV=0.001):
    """ Calculate stability matrices (conductance and current) of QD-system.

    Arguments:
    system -- qmeq.Builder object representing QD-system
    vlst -- list of bias voltages
    vglst -- list of gate voltages
    delta_eps -- Maybe remove since parallel dot has it has attribute?
    dV -- step size for numerical derivative

    Returns:
    stab -- current matrix
    stab_cond -- conductance matrix
    """
    vpnt, vgpnt = vlst.shape[0], vglst.shape[0]
    stab = np.zeros((vpnt, vgpnt))
    stab_cond = stab.copy()

    for j1 in range(vgpnt):
        system.change(hsingle={(0, 0): -vglst[j1] + delta_eps,
                               (1, 1): -vglst[j1] - delta_eps})
        system.solve(masterq=False)  # diagonalizes, but doesn't solve
        for j2 in range(vpnt):
            system.change(mulst={0: vlst[j2]/2, 1: -vlst[j2]/2})
            system.solve(qdq=False)  # solves, but doesn't diagonalize
            stab[j1, j2] = system.current[0]

            system.add(mulst={0: dV/2, 1: -dV/2})
            system.solve(qdq=False)
            stab_cond[j1, j2] = (system.current[0] - stab[j1, j2])/dV
    return stab, stab_cond


def stab_plot(stab, stab_cond, vlst, vglst, U, gam):
    """Plots stability diagrams.

    Arguments:
    stab -- current matrix from stab_calc
    stab_cond -- conductance matrix from stab_calc
    vlst -- list of bias voltages
    vglst -- list of gate voltages
    U -- coulomb energy (to get correct scaling) Maybe not needed?
    gam -- gamma of system
    """

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
    """ Plots the stability diagrams of the system for standard parameters.

    Arguments:
    system -- qmeq.Builder object representing QD-system
    delta_eps -- maybe not needed?
    gamma -- gamma of system
    parameters -- 'stephanie' or 'paper'
    """
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


def plot_spectrum(eigvals):
    """ Plots the eigenvalues in the complex plane.

    Arguments:
    eigvals -- eigenvalues to be plotted
    """
    im_parts = [eigval.imag for eigval in eigvals]
    re_parts = [eigval.real for eigval in eigvals]

    fig, ax = plt.subplots()
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    ax.grid(True, which='both')
    ax.set_axisbelow(True)
    ax.axhline(y=0, color='k', zorder=1)
    ax.axvline(x=0, color='k', zorder=1)
    ax.scatter(re_parts, im_parts, c=colors, s=100, zorder=2)
    f_size = 15
    ax.set_xlabel(r'Re$\lambda$', fontsize=f_size)
    ax.set_ylabel(r'Im$\lambda$', fontsize=f_size)
    # plt.savefig('../spectrum.png', dpi=300)
    plt.show()


def calc_tuning(delta_epsilons, system, lamb_shift):
    """ Calculates the tuning process of eigenvalues when changing delta_eps.

    Arguments:
    delta_epsilons -- vector of delta_epsilons for tuning
    system -- ParallelDot object

    Returns:
    eigs -- matrix of eigenvalues (columns) for different delta_epsilons
            (rows)
    """
    system_size = system.kern.shape[0]
    eigs = np.zeros((len(delta_epsilons), system_size), dtype=complex)

    for i, delta_e in enumerate(delta_epsilons):
        system.change_delta_eps(delta_e)
        system.solve(lamb_shift=lamb_shift)  # masterq=False??
        eigs[i, :] = system.eigvals

    return eigs


def plot_tuning(eigs, delta_epsilons):
    """ Plots the tuning process of the eigenvalues when changing delta_eps.

    Parameters:
    eigs -- matrix of eigenvalues at different delta_epsilon from calc_tuning
    delta_epsilons -- vector of delta_epsilons
    """
    system_size = eigs.shape[1]
    fig, (ax_real, ax_imag, ax_spec) = plt.subplots(1, 3, figsize=(10, 4))
    for eig_index in range(system_size):
        ax_real.plot(delta_epsilons, np.real(eigs[:, eig_index]))
        ax_imag.plot(delta_epsilons, np.imag(eigs[:, eig_index]))

        im_parts = [eigval.imag for eigval in eigs[:, eig_index]]
        re_parts = [eigval.real for eigval in eigs[:, eig_index]]
        # scatter plot where the size changes over the delta_epsilons
        ax_spec.scatter(re_parts, im_parts,
                        s=np.linspace(5, 12, len(delta_epsilons)))
    # ax_real.legend(np.arange(system_size))
    # ax_imag.legend(np.arange(system_size))
    fs = 15
    ax_real.set_xlabel(r'$\delta\epsilon$', fontsize=fs)
    ax_imag.set_xlabel(r'$\delta\epsilon$', fontsize=fs)
    ax_spec.set_xlabel(r'Re$\lambda$', fontsize=fs)
    ax_real.set_ylabel(r'Re$\lambda$', fontsize=fs)
    ax_imag.set_ylabel(r'Im$\lambda$', fontsize=fs)
    ax_spec.set_ylabel(r'Im$\lambda$', fontsize=fs)
    plt.tight_layout()
    # plt.savefig('../tuning_Lind_300.png', dpi=400)
    plt.show()


def calc_delta_eps_at_exc_point(delta_epsilons, eigs1, eigs2):
    """ Calculates delta_epsilon at eigenvalue crossing.

    Parameters:
    delta_epsilons -- vector of delta_epsilons
    eigs1 -- vector of 1st eigenvalue at different delta_epsilons
    eigs2 -- vector of 2st eigenvalue at different delta_epsilons

    Returns:
    delta_epsilon -- the value in delta_epsilons which is closest to the
                     crossing

    Warning: Must check if there is an exceptional point in delta_epsilons
             first.
    """
    min_mag_index = 0
    min_mag = float('inf')
    for i, (eig1_i, eig2_i) in enumerate(zip(eigs1, eigs2)):
        current_mag = abs(eig1_i-eig2_i)
        if current_mag < min_mag:
            min_mag_index = i
            min_mag = current_mag

    return delta_epsilons[min_mag_index]


def plot_int_vs_diag(system, t_vec):
    """Plots the norm-difference between using diagonalization and integration.
    """
    # Trace distance of the vectorized matrices?
    res_diag = np.array([list(system.dens_matrix_evo(t)) for t in t_vec])

    def rhs(t, y):
        L = system.kern
        return L@y

    res_int = solve_ivp(rhs, (0, t_vec[-1]), system.rho_0, t_eval=t_vec).y.T
    norms = 1/2 * np.array(np.linalg.norm(res_diag - res_int, axis=1))
    fig, ax = plt.subplots(1, 1)
    ax.plot(t_vec, norms)
    fs = 13
    ax.set_xlabel(r'$t$', fontsize=fs)
    ax.set_ylabel(r'$||\rho_{diag} - \rho_{int}||$', fontsize=fs)
    # plt.savefig('../intvssteady.png', dpi=400)
    plt.show()


def print_trace_evo(system, t_vec):
    res_diag = np.array([list(system.dens_matrix_evo(t)) for t in t_vec])
    traces = [np.trace(system._vector2matrix(res_diag[i, :]))
              for i in range(len(t_vec))]
    print(np.array_str(np.array(traces), precision=4, suppress_small=True))


def trace_distance(vec1, vec2):
    pass


def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(line.split()) + r'\\' for line in lines]
    rv += [r'\end{bmatrix}']
    return '\n'.join(rv)


if __name__ == '__main__':
    gamma = 1
    delta_eps = gamma*0.29587174348697
    # delta_eps = gamma*0.5
    delta_t = gamma*1e-6
    v_bias = 30*gamma

    #           upper->L, upper->R, lower->L, lower->R
    d_vec = np.array([-1, 1, 1, -1])
    rho_0 = np.array([0.3, 0.2, 0.2, 0.3, 0.2, -0.2], dtype=complex)
    parallel_dots = ParallelDots(gamma, delta_eps, delta_t, d_vec, rho_0,
                                 'pyLindblad', parameters='stephanie',
                                 v_bias=v_bias)
    parallel_dots.solve(lamb_shift=False)
    tvec = np.linspace(0, 10, 20)
    print_trace_evo(parallel_dots, tvec)

