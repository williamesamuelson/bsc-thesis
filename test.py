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
    spec_fmt = {'color': 'blue', 'ls': 'none', 'marker': '.', 'markersize': 15}
    ax.plot(re_parts, im_parts, **spec_fmt)
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.show(block=False)


def calc_tuning(delta_epsilons, system, sort=True):
    """ Calculates the tuning process of eigenvalues when changing delta_eps.

    Arguments:
    delta_epsilons -- vector of delta_epsilons for tuning
    system -- ParallelDot object
    sort -- Determines if the eigenvalues are sorted or not. Sorting causes
            jumping at crossings, but fixes random jumpings caused by numpy.

    Returns:
    eigs -- matrix of eigenvalues (columns) for different delta_epsilons
            (rows)
    """
    system_size = system.kern.shape[0]
    eigs = np.zeros((len(delta_epsilons), system_size), dtype=complex)

    for i, delta_e in enumerate(delta_epsilons):
        system.change_delta_eps(delta_e)
        system.solve()
        current_eigvals = np.linalg.eigvals(system.kern)
        if sort:
            eigs[i, :] = np.sort(current_eigvals)
        else:
            eigs[i, :] = current_eigvals

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
                        s=np.linspace(2, 8, len(delta_epsilons)))
    plt.show()
    ax_real.legend(np.arange(system_size))
    ax_imag.legend(np.arange(system_size))


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


def check_if_exc_point(system, eigenval_tol=0.1, eigenvec_tol=0.1):
    """ Checks if system is at an exceptional point.


    Parameters:
    system -- the QD system (qmeq.Builder object)
    eigenval_tol -- tolerance for eigenvalue merge
    eigenvec_tol -- tolerance for eigenvector merge

    Returns:
    indices -- a pair of indices which describe which eigenvectors correspond
               to exceptional point. The indices correspond to the ones in
               system.eigenvalues. Empty if no exceptional point.
    """
    eigenvals, eigenvecs = system.eigvals, system.r_eigvecs

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

    # Make copy of potential_indices and loop over it ([:]). This way one can
    # remove elements in the original list when iterating over it
    for i, j in potential_indices[:]:
        distance = np.linalg.norm(eigenvecs[:, i]-eigenvecs[:, j])
        if distance > eigenvec_tol:
            # remove from potential_indices
            potential_indices.remove((i, j))

    # returns empty list if no exceptional point
    return potential_indices


def print_orth_matrix(system):
    # is the vectorized scalar product wl.H*wr?
    wr = system.r_eigvecs
    wl = system.l_eigvecs
    dots = np.array([[np.vdot(wl[:, i], wr[:, j]) for i in range(len(wr))]
                    for j in range(len(wr))])
    print(np.array_str(dots, precision=1, suppress_small=True))


def plot_int_vs_diag(system, t_vec):
    res_diag = np.array([list(system.dens_matrix_evo(t)) for t in t_vec])

    def rhs(t, y):
        L = system.kern
        return L@y

    res_int = solve_ivp(rhs, (0, t_vec[-1]), rho_0, t_eval=t_vec).y.T
    norms = 1/2 * np.array(np.linalg.norm(res_diag - res_int, axis=1))
    fig, ax = plt.subplots(1, 1)
    ax.plot(t_vec, norms)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$||\rho_{diag} - \rho_{int}||$')
    plt.show()
    print(sum(res_int[-1, :-2]))
    print(sum(res_diag[-1, :-2]))
    



if __name__ == '__main__':
    gamma = 1
    delta_eps = gamma*0.29587174348697
    # delta_eps = gamma*1
    delta_t = gamma*1e-3
    v_bias = 30*gamma

    #           upper->L, upper->R, lower->L, lower->R
    d_vec = np.array([-1, 1, 1, -1])
    rho_0 = np.array([0.3, 0.2, 0.2, 0.3, 0.1, 0.1], dtype=complex)
    parallel_dots = ParallelDots(gamma, delta_eps, delta_t, d_vec, rho_0,
                                 'pyLindblad', parameters='stephanie',
                                 v_bias=v_bias)
    parallel_dots.solve()
    print(parallel_dots.check_if_exc_point())
    t_vec = np.linspace(0, 5, 50)
    plot_int_vs_diag(parallel_dots, t_vec) 
