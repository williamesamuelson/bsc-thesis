"""Module containing lamb shifted Lindbladian.

Author: Stephanie Matern
"""

import numpy as np
from scipy.special import digamma
import qmeq


# Fermi Dirac distribtuion
def FermiDirac(E, mu, T, hole=False):
    arg = (E - mu)/T
    # arg [np .abs(arg)>100]=100.
    if hole:
        return 1 - 1./(np.exp(arg)+1)
    else:
        return 1./(np.exp(arg)+1)


def vec_H(matrix, dm=0):
    """Vectorisation of the free evolution part [H,rho] -> H' vec(rho), where
    H is (4,4) and H' = (6,6)
    """
    identity = np.identity(4)

    if dm == 0:
        res = np.kron(identity, matrix)
    else:
        res = np.kron(np.conjugate(np.transpose(matrix)), identity)

    # only take needed elements
    res = np.delete(res, [1, 2, 3, 4, 7, 8, 11, 12, 13, 14], axis=0)
    res = np.delete(res, [1, 2, 3, 4, 7, 8, 11, 12, 13, 14], axis=1)

    return res


def vec_dissipator(matrix1, matrix2, dm=0):
    """Vectorisation of disspation part D[rho] -> D' vec(rho) with D'.shape = (6,6)
    """

    identity = np.identity(4)

    if dm == 0:
        product = np.dot(np.conjugate(np.transpose(matrix2)), np.conjugate(np.transpose(matrix1)))
        res = np.kron(product, identity)
    elif dm == 1:
        res = np.kron(np.conjugate(np.transpose(matrix2)), matrix1)
    else:
        product = np.dot(matrix1, matrix2)
        res = np.kron(identity, product)

    # only take needed elements
    res = np.delete(res,[1,2,3,4,7,8,11,12,13,14],axis=0)
    res = np.delete(res,[1,2,3,4,7,8,11,12,13,14],axis=1)

    return res


def calc_Lindblad_kernel(system, swap=False):

    # -------------------------
    # input parameters:

    # eps1 and eps2: dot energies for double dot system
    # hop_arr: (4,) array with hopping amplitudes t_is for dot i and lead s
    # muL, TL, muR, TR: chemical potentials and temperatures for left and right lead
    # Uval : Coulomb interaction
    # -------------------------
    #
    # output parameters:

    # Lindblad kernel: (6,6) array, some ordering as qmeq, ie diagonal elements, the real and imaginiary part 
    # stationary state: (6,) array representing stationary state of density matrix
    # -------------------------
    # -------------------------

    eps1, eps2 = list(system.hsingle.values())
    hop_arr = list(system.tleads.values())
    muL, muR = list(system.mulst)
    TL, TR = list(system.tlst)
    Uval = list(system.coulomb.values())[0]

    t1L = hop_arr[0]
    t2L = hop_arr[2]

    t1R = hop_arr[1]
    t2R = hop_arr[3]

    # build Lindblad operators
    #############################
    # move outside and insert parameters instead so that I can use it for
    # current
    def build_jump_operators():

        # dot 1 -------------------------
        L1 = np.zeros((4, 4))
        L1[1, 0] = np.sqrt(FermiDirac(eps1, muL, TL))
        L1[3, 2] = -np.sqrt(FermiDirac(eps1 + Uval, muL, TL))

        L2 = np.zeros((4, 4))
        L2[0, 1] = np.sqrt(1 - FermiDirac(eps1, muL, TL))
        L2[2, 3] = -np.sqrt(1 - FermiDirac(eps1 + Uval, muL, TL))

        L3 = np.zeros((4, 4))
        L3[1, 0] = np.sqrt(FermiDirac(eps1, muR, TR))
        L3[3, 2] = -np.sqrt(FermiDirac(eps1 + Uval, muR, TR))

        L4 = np.zeros((4, 4))
        L4[0, 1] = np.sqrt(1 - FermiDirac(eps1, muR, TR))
        L4[2, 3] = -np.sqrt(1 - FermiDirac(eps1 + Uval, muR, TR))
        # --------------------------------

        # dot 2 -------------------------
        L5 = np.zeros((4,4))
        L5[2,0] = np.sqrt(FermiDirac(eps2, muL, TL))
        L5[3,1] = np.sqrt(FermiDirac(eps2 + Uval, muL, TL))

        L6 = np.zeros((4,4))
        L6[0,2] = np.sqrt(1 - FermiDirac(eps2, muL, TL))
        L6[1,3] = np.sqrt(1 - FermiDirac(eps2 + Uval, muL, TL))

        L7 = np.zeros((4,4))
        L7[2,0] = np.sqrt(FermiDirac(eps2, muR, TR))
        L7[3,1] = np.sqrt(FermiDirac(eps2 + Uval, muR, TR))

        L8 = np.zeros((4,4))
        L8[0,2] = np.sqrt(1 - FermiDirac(eps2, muR, TR))
        L8[1,3] = np.sqrt(1 - FermiDirac(eps2 + Uval, muR, TR))
        #--------------------------------

        # combine processes

        La = t1L * L1 + t2L * L5
        Lb = t1L * L2 + t2L * L6
        Lc = t1R * L3 + t2R * L7
        Ld = t1R * L4 + t2R * L8

        # write all jump operators in tensor

        L_tensor = np.zeros((4,4,4))

        L_tensor[0,:,:] = La
        L_tensor[1,:,:] = Lb
        L_tensor[2,:,:] = Lc
        L_tensor[3,:,:] = Ld

        return L_tensor
    #############################

    # compare dissipation with qmeq, sanity check
    #############################
    def get_qmeq_kernel():
        
        # builds qmeq system and returns kernel
        
        mulst = {0:muL, 1:muR}
        n = 2
        nleads = 2
        tlst = {0:TL, 1:TR}
        U = {(0,1,1,0):Uval}
        h_asym = {(0,0): eps1, (1,1): eps2}
        tleads_asym = {(0, 0):t1L, (1, 0):t1R, (0, 1):t2L, (1, 1):t2R}
        system = qmeq.Builder(nsingle=n, hsingle=h_asym, coulomb=U, nleads=nleads, kerntype= 'pyLindblad',
                              mulst=mulst, tlst=tlst, tleads=tleads_asym, dband=1e10, )
        system.solve(currentq = False)
        
        return system.kern
    #############################
        
    # build Lambshift ### combined operators such that they are hermitian, not used here 
    #############################
    def build_Lambshift_herX():
        
        # returns Lambshift as (4,4) matrix
    
        D = 1e10  #value of dband
        t_rs = hop_arr.reshape(2,2)
        
        
        # Principle value integrals
        #############################
        def S11(t1, t2, e, mu, T):
            
            # particle
            p_int1 =   -  (t1 * t2 * (np.real(digamma(0.5 + 1j * (e - mu)/(2 * np.pi * T)))
                                  - np.log(D/(2 * np.pi * T))))
            # hole
            p_int2 =   (t1 * t2 * (np.real(digamma(0.5 + 1j * (- e -  mu)/(2 * np.pi * T)))
                                - np.log(D/(2 * np.pi * T)) + np.log(np.abs((D-mu)/(-D-mu)))))

            return   1j * (p_int1 - p_int2)

        def S12(t1, t2, e, mu, T):

            p_int1 =   - (t1 * t2 * (np.real(digamma(0.5 + 1j * (e -  mu)/(2 * np.pi * T)))
                                  - np.log(D/(2 * np.pi * T))))

            p_int2 =   (t1 * t2 * (np.real(digamma(0.5 + 1j * (- e - mu)/(2 * np.pi * T)))
                                - np.log(D/(2 * np.pi * T)) + np.log(abs((D-mu)/(-D-mu)))))

            return   -  (p_int1 + p_int2)
        #############################

        # define operator is eigenbasis |a>, |b>, |c>, |d>

        a = np.array([1,0,0,0])
        b = np.array([0,1,0,0])
        c = np.array([0,0,1,0])
        d = np.array([0,0,0,1])

        ab = np.outer(a,b)
        ac = np.outer(a,c)

        ba = np.outer(b,a)
        bd = np.outer(b,d)

        ca = np.outer(c,a)
        cd = np.outer(c,d)

        db = np.outer(d,b)
        dc = np.outer(d,c)


        x11 = 0.5 * (ba + dc +  ab + cd)
        x12 = 0.5 * (ca - db + ac - bd)

        x21 = -0.5j * (ba + dc -  ab - cd) 
        x22 = -0.5j * (ca - db - ac + bd)
    
        x_arr = np.array([[x11,x12],[x21,x22]])

        LS = np.zeros((4,4)) + 0j

        ee = np.array([0,eps1, eps2, eps1 + eps2 + Uval])

        for m in range(4):
            for l in range(4):
                for alpha in range(2):
                    for beta in range(2):
                        for j in range(2):
                            for k in range(2):
                                for n in range(4):
                                    omega_mn = (ee[n] - ee[m])
                                    omega_nl = (ee[n] - ee[l])

                                    t1 = t_rs[j]
                                    t2 = t_rs[k]

                                    if alpha == beta:
                                        SL_mn = S11(t1[0], t2[0], omega_mn, muL, TL)
                                        SR_mn = S11(t1[1], t2[1], omega_mn, muR, TR)
                                        SL_nl = S11(t1[0], t2[0], omega_nl, muL, TL)
                                        SR_nl = S11(t1[1], t2[1], omega_nl, muR, TR)
                                        S_tot = 0.25 * (SL_mn + SR_mn + SL_nl + SR_nl)
                                        
                                    else:
                                        if alpha > beta:
                                            pre = -1
                                        else:
                                            pre = 1
                                        SL_mn = S12(t1[0], t2[0], omega_mn, muL, TL)
                                        SR_mn = S12(t1[1], t2[1], omega_mn, muR, TR)
                                        SL_nl = S12(t1[0], t2[0], omega_nl, muL, TL)
                                        SR_nl = S12(t1[1], t2[1], omega_nl, muR, TR)
                                        S_tot = pre * 0.25 * (SL_mn + SR_mn + SL_nl + SR_nl)

                                    LS[m,l] +=0.5 *  x_arr[alpha,j][m,n] * x_arr[beta,k][n,l] * S_tot

        return  4j * LS
    
    def build_Lambshift():
    
        ee = np.array([0,eps1, eps2, eps1 + eps2 + Uval])

        def princ_val(i,j, mu, temp, particle = True):

            D = 1e10
            epsilon = (ee[i] - ee[j])

            if particle:
                res = - (np.real(digamma(0.5 + 1j * (epsilon - mu)/(2 * np.pi * temp)))  - np.log(D/(2 * np.pi * temp)))
            else:
                res = - (np.real(digamma(0.5 + 1j * (- epsilon - mu)/(2 * np.pi * temp)))
                        - np.log(D/(2 * np.pi * temp)) + np.log(abs((D-mu)/(-D-mu))))

            return res


        def get_S(a,b, i, j):

            #particle
            int_pL = princ_val(i,j, muL, TL )
            int_pR = princ_val(i,j, muR, TR )
    
            #hole
            int_hL = princ_val(i,j, muL, TL, False)
            int_hR = princ_val(i,j, muR, TR, False)

            hopping_left = np.zeros((4,4))
            hopping_right = np.zeros((4,4))

            hopping_left[0,2] =  hop_arr[0] ** 2
            hopping_left[0,3] =  hop_arr[0] * hop_arr[2]

            hopping_left[1,2] =  hop_arr[0] * hop_arr[2]
            hopping_left[1,3] =  hop_arr[2] ** 2 

            hopping_left = hopping_left + np.transpose(hopping_left)


            hopping_right[0,2] =  hop_arr[1] ** 2
            hopping_right[0,3] =  hop_arr[1] * hop_arr[3]

            hopping_right[1,2] =  hop_arr[1] * hop_arr[3]
            hopping_right[1,3] =  hop_arr[3] ** 2 

            hopping_right = hopping_right + np.transpose(hopping_right)

            if a < b:
                #print(hopping_left[a,b] * int_hL + hopping_right[a,b] * int_hR) 
                return 1j * (hopping_left[a,b] * int_hL + hopping_right[a,b] * int_hR)
            if a > b:
                #print(hopping_left[a,b] * int_pL + hopping_right[a,b] * int_pR)
                return 1j * (hopping_left[a,b] * int_pL + hopping_right[a,b] * int_pR)
            else:
                #print(0)
                return 0


        # define operator is eigenbasis |a>, |b>, |c>, |d>

        a = np.array([1,0,0,0])
        b = np.array([0,1,0,0])
        c = np.array([0,0,1,0])
        d = np.array([0,0,0,1])

        ab = np.outer(a,b)
        ac = np.outer(a,c)

        ba = np.outer(b,a)
        bd = np.outer(b,d)

        ca = np.outer(c,a)
        cd = np.outer(c,d)

        db = np.outer(d,b)
        dc = np.outer(d,c)

        x1 = (ba - dc)
        x2 = (ca + db)

        x3 = (ab - cd) 
        x4 = (ac + bd)

        #x1 = ab
        #x2 = ac
        #x3 = -bd
        #x4 = cd

        x_arr = np.array([x1,x2,x3,x4])

        LS = np.zeros((4,4)) + 0j


        for m in range(4):
            for l in range(4):
                for alpha in range(4):
                    for beta in range(4):
                        for n in range(4):

                            LS[m,l] += (np.conjugate(x_arr[alpha])[m,n] * x_arr[beta][n,l] * 
                                       0.5 * (get_S(alpha,beta,n,m) + get_S(alpha,beta, n,l)) )

        return 1j * LS

    # build (6,6) Lindblad kernel including Lambshift

    mykern = np.zeros((6, 6)) + 0j

    # dissipation part
    jump = build_jump_operators()
    for i in range(jump.shape[0]):
        term1 = vec_dissipator(jump[i],np.transpose(np.conjugate(jump[i])), dm=1)
        term2 = - 0.5 * vec_dissipator(np.transpose(np.conjugate(jump[i])), jump[i], dm=0)
        term3 = - 0.5 * vec_dissipator(np.transpose(np.conjugate(jump[i])), jump[i], dm=2)
        mykern += 2 * np.pi * (term1 + term2 + term3)

    # free evolution of Hamiltonian
    eigene = np.array([0,eps1, eps2, eps1 + eps2 + Uval])
    ham = np.diag(eigene)
    free_evo = 1j * (vec_H(ham, dm = 1) - vec_H(ham, dm = 0) )
    mykern += free_evo

    mykern_copy = mykern

    # transform kernel into qmeq basis

    Utrans = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[0,0,0.5,0.5,0,0],[0,0,0.5j, -0.5j,0,0]])

    mykern = Utrans @ mykern @ np.linalg.inv(Utrans)
    
    # compare with qmeq before adding lambshift
    # qmeq_kern = get_qmeq_kernel()
    qmeq_kern = system.kern
    if np.allclose(np.real(mykern),qmeq_kern) == False:
        print('Warning: qmeq and constructed Lindbladian not the same -- change state ordering')
        #print(qmeq_kern - mykern)
        #print(e1)
        #print(e2)
        Utrans2 = np.array([[1,0,0,0,0,0],[0,0,0,0,1,0],[0,1,0,0,0,0],[0,0,0,0,0,1],[0,0,0.5,0.5,0,0],[0,0,-0.5j, 0.5j,0,0]])
        mykern_copy = Utrans2 @ mykern @ np.linalg.inv(Utrans2)
        if np.allclose(np.real(mykern_copy),qmeq_kern) == False:
            print('Warning: qmeq and constructed Lindbladian still not the same')
            print(np.array_str(qmeq_kern - mykern_copy, precision=1))
            print(eps1)
            print(eps2)
        else:
            mykern = mykern_copy
            Utrans = Utrans2

        
    # add lambshift 
    H_lambshift = build_Lambshift()
    #H_lambshift = build_Lambshift_herX()
    
    #sanity check if H_lambshift is hermitian
    var = np.allclose(H_lambshift, np.conjugate(np.transpose(H_lambshift)))
    if var == False:
        print('Warning: H_LS not Hermitian')
        print(H_lambshift)
        
    # vectorise Lambshift and add to kernel
    lamb_vec =  1j * (vec_H(H_lambshift, dm = 1) - vec_H(H_lambshift, dm = 0) )
    mykern +=  Utrans @ lamb_vec @np.linalg.inv(Utrans)
    
    # calculate stationary state
    
    # initial = np.array([0.25,0.25,0.25,0.25,0,0]) # some vector needed
    # 
    # ew, evl, evr = eig(mykern, left = True) 
    # 
    # idx = ew.argsort()[::-1]   
    # ew = ew[idx]
    # evl = evl[:,idx]
    # evr = evr[:,idx] 

    # mystst = 1/np.dot(np.conjugate(evl[:,0]),evr[:,0]) * np.dot(np.conjugate(evl[:,0]), initial) * evr[:,0]

    return mykern
    #return mykern, mystst
