import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from parallel_dots import ParallelDots
import pickle
from exceptional_point import ExceptionalPoint


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
    lamb_shift -- Includes lamb_shift

    Returns:
    eigs -- matrix of eigenvalues (columns) for different delta_epsilons
            (rows)
    """
    system_size = system.kern.shape[0]
    eigs = np.zeros((len(delta_epsilons), system_size), dtype=complex)

    for i, delta_e in enumerate(delta_epsilons):
        system.change_delta_eps(delta_e)
        system.solve(lamb_shift=lamb_shift, currentq=False)  # masterq=False??
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
    min_mag_index -- the index in delta_epsilons/eigs of the best value

    Warning: Must check if there is an exceptional point in delta_epsilons
             first.
    """
    min_mag_index = 0
    min_mag = float('inf')
    for i, (eig1_i, eig2_i) in enumerate(zip(eigs1, eigs2)):
        current_mag = np.abs(eig1_i-eig2_i)
        if current_mag < min_mag:
            min_mag_index = i
            min_mag = current_mag

    print(min_mag)
    print(min_mag_index)
    return delta_epsilons[min_mag_index], min_mag_index


def optimize_ep(system, init_guess, init_range, tolerance,
                ind1, ind2, lamb_shift):
    eig1 = system.eigvals[ind1]
    eig2 = system.eigvals[ind2]
    linspace_len = 5
    distance = np.abs(eig1 - eig2)
    rang_e = init_range
    current_d_eps = init_guess
    old_ind = linspace_len
    repeat = 0
    while(distance > tolerance):
        if repeat > 9:
            break
        delta_epsilons = current_d_eps + np.linspace(-rang_e, rang_e,
                                                     linspace_len)
        eigs = calc_tuning(delta_epsilons, system, lamb_shift)
        current_d_eps, index = calc_delta_eps_at_exc_point(delta_epsilons,
                                                           eigs[:, ind1],
                                                           eigs[:, ind2])
        eig1 = eigs[index, ind1]
        eig2 = eigs[index, ind2]
        distance = np.abs(eig1 - eig2)
        print(rang_e)
        print(delta_epsilons)
        print('\n')
        if not (index == linspace_len-1 or index == 0):
            rang_e /= 2
        if old_ind == index:
            repeat += 1
        else:
            repeat = 0
        old_ind = index

    system.change_delta_eps(current_d_eps)
    system.solve(lamb_shift=lamb_shift)
    return current_d_eps


def plot_int_vs_diag(system, t_vec, rho_0, ep=None):
    """Plots the norm-difference between using diagonalization and integration.

    Parameters:
    system -- ParallelDot object
    t_vec -- vector of points in time
    rho_0 -- initial density matrix
    ep -- ExceptionalPoint object
    """
    # Trace distance of the vectorized matrices?
    res_diag = calc_dens_evo(system, t_vec, rho_0, ep)

    def rhs(t, y):
        L = system.kern
        return L@y

    res_int = solve_ivp(rhs, (0, t_vec[-1]), rho_0, t_eval=t_vec).y.T
    norms = 1/2 * np.array(np.linalg.norm(res_diag - res_int, axis=1))
    fig, ax = plt.subplots(1, 1)
    ax.plot(t_vec, norms)
    fs = 13
    ax.set_xlabel(r'$t$', fontsize=fs)
    ax.set_ylabel(r'$||\rho_{jord} - \rho_{int}||$', fontsize=fs)
    # plt.savefig('../intvsjord_lshift_rho3.png', dpi=400, bbox_inches='tight')
    plt.show()


def print_trace_evo(system, t_vec, rho_0, ep=None):
    """Prints the trace of rho over time.

    Parameters:
    system -- ParallelDot object
    t_vec -- vector of points in time
    rho_0 -- initial density matrix
    ep -- ExceptionalPoint object
    """
    res_diag = calc_dens_evo(system, t_vec, rho_0, ep)
    traces = [sum(res_diag[t, 0:4]) for t in range(len(t_vec))]
    print(np.array_str(np.array(traces), precision=6, suppress_small=True))


def calc_dens_evo(system, t_vec, rho_0, ep=None):
    """Calculates the density matrix evolution.

    Parameters:
    system -- ParallelDot object
    t_vec -- vector of points in time
    rho_0 -- initial density matrix
    ep -- ExceptionalPoint object

    Returns:
    res -- len(t_vec)x6 matrix of density matrices in vector form at different
           times
    """
    if ep is None:
        res = np.array([list(system.dens_matrix_evo(t, rho_0)) for t in t_vec])
    else:
        res = np.array([list(system.dens_matrix_evo_ep(t, rho_0, ep))
                        for t in t_vec])
    return res


def plot_current(system, t_vec, rho_0, direction, axis, ep=None):
    """Plots the current over time normalized by steady state current.

    Parameters:
    system -- ParallelDot object
    t_vec -- vector of points in time
    rho_0 -- initial density matrix
    direction -- direction of current (left/right)
    axis -- axes object for plot
    ep -- ExceptionalPoint object
    """

    res_diag = calc_dens_evo(system, t_vec, rho_0, ep)
    res_curr = [system.calc_current(res_diag[i, :], direction)
                for i in range(len(t_vec))]
    ss_curr = system.calc_ss_current(rho_0, direction)
    if not np.allclose(np.imag(res_curr), 0) or ss_curr.imag > 1e-6:
        raise Exception('Imaginary parts in current')

    axis.plot(t_vec, np.real(res_curr)/ss_curr.real)
    axis.set_xlabel(r'Time $(t)$', fontsize=15)
    axis.set_ylabel(r'$I(t)/I_{ss}$', fontsize=20)
    axis.axhline(y=1, color='black', zorder=0)


def plot_current_ep_vs_nonep(system, t_vec, rho_0, direction, d_epsilons,
                             l_shift, ep):
    fig, ax = plt.subplots()
    exc_points = len(d_epsilons)*[None]
    exc_points[0] = ep
    for exc_point, d_epsilon in zip(exc_points, d_epsilons):
        parallel_dots.change_delta_eps(d_epsilon)
        parallel_dots.solve(lamb_shift=l_shift)
        plot_current(parallel_dots, t_vec, rho_0, direction, ax, ep=exc_point)
    ax.legend(['EP', 'non-EP', 'non-EP2'], fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    # plt.savefig('../current_epvsnonep.png', dpi=400, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    GAMMA = 1
    pickle_off = open('../d_eps_ep_lamb_shift.txt', 'rb')
    # using v_b = 350*GAMMA and delta_t = 1e-6
    DELTA_EPS = pickle.load(pickle_off) * GAMMA
    # DELTA_EPS = GAMMA*0.29587174348697  # for lindblad no lamb shift
    DELTA_T = GAMMA*1e-6
    V_BIAS = 350*GAMMA

    #           upper->L, upper->R, lower->L, lower->R
    d_vec = np.array([-1, 1, 1, -1])
    parallel_dots = ParallelDots(GAMMA, DELTA_EPS, DELTA_T, d_vec,
                                 'pyLindblad', parameters='stephanie',
                                 v_bias=V_BIAS)
    l_shift = True
    parallel_dots.solve(lamb_shift=l_shift)
    tvec = np.linspace(0, 10, 100)
    ep = ExceptionalPoint(parallel_dots, 'full space')
    rho_0 = ep.R[:, 0] + ep.R[:, 2]
    rho_0 /= sum(rho_0[:4])
    rho_0 = np.array([0.5, 0.2, 0.1, 0.3, 0.1, -0.5])
    plot_int_vs_diag(parallel_dots, tvec, rho_0, ep)

