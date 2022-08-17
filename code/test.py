import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from parallel_dots import ParallelDots
from exceptional_point import ExceptionalPoint
# from functions import my_floor, my_ceil


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

    fig, ax = plt.subplots(figsize=(10,3))
    colors = ['tab:blue', 'tab:blue', 'tab:orange', 'tab:blue', 'tab:blue',
              'tab:orange']
    shapes = ['o', 'o', 'x', 'o', 'o', 'x']
    text = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$',
               r'$\lambda_5$', r'$\lambda_6$']
    f_size = 20
    ax.axhline(y=0, color='k', zorder=1, label='_nolegend_')
    ax.axvline(x=0, color='k', zorder=1, label='_nolegend_')
    for i in range(len(eigvals)):
        ax.scatter(re_parts[i], im_parts[i], s=200, zorder=2, marker=shapes[i],
                   c=colors[i])
        if i == 2 or i == 5:
            ax.text(re_parts[i], im_parts[i] - 0.44, text[i], size=f_size)
        else:
            ax.text(re_parts[i] + 0.02, im_parts[i] + 0.2, text[i], size=f_size)
    ax.set_xlabel(r'Re$\lambda$', fontsize=f_size)
    ax.set_ylabel(r'Im$\lambda$', fontsize=f_size)
    ax.set_ylim([-1, 1])
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    plt.tight_layout()
    # plt.savefig('../text/figures/spectrum.png', dpi=400, bbox_inches='tight')
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
        system.solve(lamb_shift=lamb_shift, currentq=False, sort=False)  # masterq=False??
        eigs[i, :] = system.eigvals
    return eigs


def plot_tuning(eigs, delta_epsilons, indices):
    """ Plots the tuning process of the eigenvalues when changing delta_eps.

    Parameters:
    eigs -- matrix of eigenvalues at different delta_epsilon from calc_tuning
    delta_epsilons -- vector of delta_epsilons
    indices -- which eigvals to plot
    """
    # system_size = eigs.shape[1]
    fig, (ax_real, ax_imag) = plt.subplots(1, 2, figsize=(10, 4))
    linestyles = ['solid', 'dashed']
    linewidth = 4
    for i, eig_index in enumerate(indices):
        ax_real.plot(delta_epsilons, np.real(eigs[:, eig_index]),
                     ls=linestyles[i], linewidth=linewidth)
        ax_imag.plot(delta_epsilons, np.imag(eigs[:, eig_index]),
                     ls=linestyles[i], linewidth=linewidth)

        # im_parts = [eigval.imag for eigval in eigs[:, eig_index]]
        # re_parts = [eigval.real for eigval in eigs[:, eig_index]]
        # scatter plot where the size changes over the delta_epsilons
        # ax_spec.scatter(re_parts, im_parts,
        #                 s=np.linspace(5, 12, len(delta_epsilons)))
    # ax_real.legend(np.arange(system_size))
    # ax_imag.legend(np.arange(system_size))
    fs = 20
    ax_real.set_xlabel(r'$\delta\epsilon/\Gamma$', fontsize=fs)
    ax_imag.set_xlabel(r'$\delta\epsilon/\Gamma$', fontsize=fs)
    # ax_spec.set_xlabel(r'Re$\lambda$', fontsize=fs)
    ax_real.set_ylabel(r'Re$\lambda/\Gamma$', fontsize=fs)
    ax_imag.set_ylabel(r'Im$\lambda/\Gamma$', fontsize=fs)
    ax_imag.tick_params(axis='both', which='major', labelsize=13)
    ax_imag.tick_params(axis='both', which='minor', labelsize=10)
    ax_real.tick_params(axis='both', which='major', labelsize=13)
    ax_real.tick_params(axis='both', which='minor', labelsize=10)
    ax_real.legend([r'$\lambda_5$', r'$\lambda_6$'], fontsize=16)
    # ax_spec.set_ylabel(r'Im$\lambda$', fontsize=fs)
    plt.tight_layout()
    # plt.savefig('../text/figures/tuning.png', dpi=400)
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
    while distance > tolerance:
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


def calc_dens_evo(system, t_vec, rho_0, method, ep=None):
    """Calculates the density matrix evolution.

    Parameters:
    system -- ParallelDot object
    t_vec -- vector of points in time
    rho_0 -- initial density matrix
    method -- method of calculation, 'ep', 'diag' or 'num'.
    ep -- ExceptionalPoint object

    Returns:
    res -- len(t_vec)x6 matrix of density matrices in vector form at different
           times
    """
    if method == 'diag':
        res = np.array([list(system.dens_matrix_evo(t, rho_0)) for t in t_vec])
    elif method == 'ep':
        res = np.array([list(system.dens_matrix_evo_ep(t, rho_0, ep))
                        for t in t_vec])
    elif method == 'num':

        def rhs(t, y):
            L = system.kern
            return L@y

        res = solve_ivp(rhs, (0, t_vec[-1]), rho_0, t_eval=t_vec).y.T

    return res


def plot_int_vs_diag(system, t_vec, rho_0, method, ep=None):
    """Plots the norm-difference between using 'method' and integration.

    Parameters:
    system -- ParallelDot object
    t_vec -- vector of points in time
    rho_0 -- initial density matrix
    method -- method of calculation 'ep' or 'diag'
    ep -- ExceptionalPoint object
    """
    # Trace distance of the vectorized matrices?
    res_diag = calc_dens_evo(system, t_vec, rho_0, method, ep)
    res_int = calc_dens_evo(system, t_vec, rho_0, 'num')

    norms = 1/2 * np.array(np.linalg.norm(res_diag - res_int, axis=1))
    norms_int = 1/2 * np.array(np.linalg.norm(res_int, axis=1))
    fig, ax = plt.subplots(1, 1)
    ax.plot(t_vec, norms/norms_int)
    fs = 13
    ax.set_xlabel(r'$t$', fontsize=fs)
    ax.set_ylabel(r'$||\rho_{jord} - \rho_{int}||/||\rho_{int}||$',
                  fontsize=fs)
    # plt.savefig('../figures/intvsdiag_atep_rhoprime.png', dpi=400,
    #             bbox_inches='tight')
    plt.show()


def print_trace_evo(system, t_vec, rho_0, method, ep=None):
    """Prints the trace of rho over time.

    Parameters:
    system -- ParallelDot object
    t_vec -- vector of points in time
    rho_0 -- initial density matrix
    method -- method of calculation, 'ep', 'diag' or 'num'.
    ep -- ExceptionalPoint object
    """
    res_diag = calc_dens_evo(system, t_vec, rho_0, method, ep)
    traces = [sum(res_diag[t, 0:4]) for t in range(len(t_vec))]
    print(np.array_str(np.array(traces), precision=6, suppress_small=True))


def plot_current(system, t_vec, rho_0, direction, axis, plot, method,
                 linestyle, ep=None):
    """Plots the current over time.

    Parameters:
    system -- ParallelDot object
    t_vec -- vector of points in time
    rho_0 -- initial density matrix
    direction -- direction of current (left/right)
    axis -- axes object for plot
    plot -- what to plot, 'divide', 'subtract_log' or 'normal'
    method -- method of calculation, 'ep', 'diag' or 'num'.
    linestyle -- linestyle for plot
    ep -- ExceptionalPoint object, None for using dens_matrix_evo instead of
          dens_matrix_evo_ep
    """

    res_diag = calc_dens_evo(system, t_vec, rho_0, method, ep)
    res_curr = [system.calc_current(res_diag[i, :], direction)
                for i in range(len(t_vec))]
    ss_curr = system.calc_ss_current(rho_0, direction)
    if not np.allclose(np.imag(res_curr), 0) or ss_curr.imag > 1e-6:
        warnings.warn('Imaginary parts in current')

    if plot == 'divide':
        axis.plot(t_vec, np.real(res_curr)/ss_curr.real, linestyle=linestyle,
                  linewidth=4)
        axis.axhline(y=1, color='black', zorder=0, label='_nolegend_')
        axis.set_ylabel(r'$I(t)/I_{ss}$', fontsize=20)
    elif plot == 'subtract_log':
        axis.set_yscale("log", base=10)
        axis.plot(t_vec,
                  np.abs(res_curr - ss_curr)/np.abs(res_curr[0] - ss_curr),
                  linestyle=linestyle, linewidth=4)
        axis.set_ylabel(r'$|I(t) - I_{ss}|/N$', fontsize=20)
    elif plot == 'normal':
        axis.plot(t_vec, np.real(res_curr), linestyle=linestyle, linewidth=4)
        axis.set_ylabel(r'$I(t)$', fontsize=20)
    else:
        raise Exception(plot + ' is not a valid "plot" entry')
    axis.set_xlabel(r'Time $(t)$', fontsize=15)


def plot_current_ep_vs_nonep(system, t_vec, rho_0, direction, d_epsilons,
                             methods, eps, l_shift, delta_eps_ep):
    """Plots current for systems for varying parameters.

    Parameters:
    system -- ParallelDots object
    t_vec -- time vector
    rho_0 -- initial density matrix
    direction -- left/right
    d_epsilons -- vector of delta epsilons
    methods -- list of methods, 'ep', 'num' or 'diag'
    eps -- list of ExceptionalPoint objects
    l_shift -- True to use lamb_shift
    """
    fig, ax = plt.subplots()
    diffs = delta_eps_ep - d_epsilons
    leg = []
    for i, (ep, d_epsilon, method) in enumerate(zip(eps, d_epsilons, methods)):
        parallel_dots.change_delta_eps(d_epsilon)
        parallel_dots.solve(lamb_shift=l_shift)
        plot_current(parallel_dots, t_vec, rho_0, direction, ax, 'divide',
                     method, ep)
        leg.append(f'$\Delta/\delta\epsilon_{{EP}} = ${diffs[i]/delta_eps_ep: .2f}')
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.legend(leg, fontsize=14)
    # plt.savefig('../text/figures/curr_diff_de.png', dpi=400,
    #             bbox_inches='tight')
    plt.show()


def plot_current_diff_rho0(system, t_vec, rhos, consts, direction, method, ep):
    """Plots current for varying initial conditions.

    Parameters:
    system -- ParallelDots object
    t_vec -- time vector
    rhos -- list of tuples representing linear combination of right
            eigenvectors used as initial condition
    consts -- list of tuples representing constants multiplying the terms
              in the linear combination
    direction -- left/right
    method -- method of calculation
    ep -- ExceptionalPoint object
    """

    fig, axis = plt.subplots()
    leg = []
    linestyles = ['solid', 'dashed']
    for i, (rho_tuple, const_tuple) in enumerate(zip(rhos, consts)):
        rho_0 = ep.R[:, 0].copy()
        for rho_ind, const in zip(rho_tuple, const_tuple):
            rho_0 += const*ep.R[:, rho_ind]
        rho_0 /= sum(rho_0[:4])
        tot_overlap = sum(np.abs(ep.L.conj().T@rho_0))
        overlap = np.abs(np.vdot(ep.L[:, 2], rho_0))
        leg.append(f'Overlap = {np.real(overlap/tot_overlap):.2f}')
        plot_current(system, t_vec, rho_0, direction, axis, 'subtract_log',
                     method, linestyles[i], ep)

    leg = [r"$\rho_0 = \rho_{ss} + \bar{\rho}$",
           r"$\rho_0 = \rho_{ss} + \rho'$"]
    axis.legend(leg, fontsize=15)
    axis.tick_params(axis='both', which='major', labelsize=13)
    axis.tick_params(axis='both', which='minor', labelsize=10)
    axis.set_xlabel('Time ' + r'$(t)$')
    # plt.savefig('../text/figures/current_diff_rho_0.png', dpi=400,
    #             bbox_inches='tight')
    plt.show()


def plot_alpha_vs_dist(delta_eps, delta_epsilons, system, t_vec, alpha_len):
    """Plots alpha vs the distance to the exceptional point.

    alpha*rho_EP + (1-alpha)*rho_diag

    Parameters:
    delta_eps -- value of delta_epsilon at EP
    delta_epsilons -- vectors of delta_epsilons
    system -- ParallelDots object
    t_vec -- vector of time points
    alpha_len -- length of alpha vector
    """

    alphas = np.zeros((len(delta_epsilons),))
    dists = []

    for i, d_eps in enumerate(delta_epsilons):
        system.change_delta_eps(d_eps)
        system.solve(lamb_shift=l_shift)
        ep = ExceptionalPoint(system, 'full space')
        rho_0 = ep.R[:, 0] + ep.R[:, 2]
        alpha = parallel_dots.optimize_alpha(ep, rho_0, t_vec, alpha_len,
                                             'length')
        alphas[i] = alpha
        dist = np.abs(ep.eigvals[ep.indices[1]] - ep.eigvals[ep.indices[0]])
        dists.append(dist)

    # fig, (ax_eig, ax_eps) = plt.subplots(1, 2)
    fig, ax_eps = plt.subplots()
    # ax_eig.set_xscale("log")
    # ax_eps.set_xscale("log")
    # ax_eig.plot(dists, alphas, '.', markersize=10)
    pos_ind = np.flatnonzero(delta_epsilons > delta_eps)
    neg_ind = np.setdiff1d(np.arange(len(delta_epsilons)), pos_ind)
    ax_eps.plot(delta_epsilons[pos_ind] - delta_eps, alphas[pos_ind], '.',
                markersize=10, color='r')
    ax_eps.plot(delta_epsilons[neg_ind] - delta_eps, alphas[neg_ind], '.',
                markersize=10, color='c')
    # ax_eig.set_xlabel('Distance between eigenvalues', fontsize=20)
    ax_eps.set_ylabel(r'$\alpha$', fontsize=20)
    ax_eps.set_xlabel('Distance from EP', fontsize=20)
    ax_eps.tick_params(axis='both', which='major', labelsize=13)
    ax_eps.tick_params(axis='both', which='minor', labelsize=10)
    ax_eps.legend([r'$\Delta > 0$', r'$\Delta < 0$'], fontsize=15)

    # plt.savefig('../figures/alphavsdiff.png', dpi=400, bbox_inches='tight')
    plt.show()


def create_delta_eps(d_eps_ep):
    delta_epsilons = np.array([d_eps_ep])
    offsets = [3e-1*d_eps_ep]
    for offset in offsets:
        delta_epsilons = np.append(delta_epsilons, d_eps_ep + offset)
        delta_epsilons = np.append(delta_epsilons, d_eps_ep - offset)

    return delta_epsilons

def help_plot_curr_epnonep(parallel_dots, DELTA_EPS):
    t_vec = np.linspace(0, 15, 100)
    ep = ExceptionalPoint(parallel_dots, 'full space')
    #rho_0 = ep.R[:, 0] - 1*ep.R[:, 2]
    rho_0 = np.array([1, 0, 0, 0, 0, 0])
    #rho_0 /= sum(rho_0[:4])
    d_epsilons = create_delta_eps(DELTA_EPS)
    plot_current_ep_vs_nonep(parallel_dots, t_vec, rho_0, 'right', d_epsilons,
                             ['num']*len(d_epsilons), [None]*len(d_epsilons), True, DELTA_EPS)


if __name__ == '__main__':
    GAMMA = 1
    with open('d_eps_ep_lamb_shift.txt', 'rb') as f:
        # using v_b = 350*GAMMA and delta_t = 1e-6
        DELTA_EPS = pickle.load(f) * GAMMA
    # for lindblad no lamb shift
    # DELTA_EPS = GAMMA*0.29693415964199998402506253114552237
    DELTA_T = GAMMA*1e-6
    V_BIAS = 350*GAMMA
    #           upper->L, upper->R, lower->L, lower->R
    d_vec = np.array([-1, 1, 1, -1])
    parallel_dots = ParallelDots(GAMMA, DELTA_EPS, DELTA_T, d_vec,
                                 'pyLindblad', parameters='stephanie',
                                 v_bias=V_BIAS)
    l_shift = True
    parallel_dots.solve(lamb_shift=l_shift)
    plot_spectrum(parallel_dots.eigvals)
