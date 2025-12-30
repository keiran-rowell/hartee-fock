import numpy as np
import math

# Code up the integral equations
def compute_S_primitive(alpha, beta, R_A, R_B):
    """Compute the overlap integral between two primitive Gaussians"""
    #Normalisation of s-type Gaussians
    N_alpha = (2.0 * alpha / np.pi)**(0.75)
    N_beta  = (2.0 * beta  / np.pi)**(0.75)

    S_unnormalised =  (np.pi / (alpha + beta))**(3/2) * np.exp(-alpha * beta / (alpha + beta) * np.sum((R_A - R_B)**2))

    return N_alpha * N_beta * S_unnormalised

def compute_T_primitive(alpha, beta, R_A, R_B, S_prim):
    """Compute the kinetic energy integral between two primitive Gaussians"""
    reduced_exp = alpha * beta / (alpha + beta)
    return reduced_exp * (3 - 2 * reduced_exp * np.sum((R_A - R_B)**2)) * S_prim

def compute_V_nuc_primitive(alpha, beta, R_A, R_B, R_nuc):
    """
    Compute nuclear attraction integral for a single nucleus
    <chi_a | -1/|r-R_nuc| | chi_b>
    """
    # Normalization
    N_alpha = (2.0 * alpha / np.pi)**(0.75)
    N_beta = (2.0 * beta / np.pi)**(0.75)
    
    zeta = alpha + beta
    P = (alpha * R_A + beta * R_B) / zeta
    
    # Gaussian product overlap factor
    AB_sq = np.sum((R_A - R_B)**2)
    K_AB = np.exp(-alpha * beta / zeta * AB_sq)
    
    # Distance from product center to nucleus
    PC_sq = np.sum((P - R_nuc)**2)
    
    if PC_sq < 1e-10:
        V_unnormalised = 2.0 * np.pi / zeta  
    else:
        T = zeta * PC_sq
        F0 = 0.5 * np.sqrt(np.pi / T) * math.erf(np.sqrt(T))
        V_unnormalised = 2.0 * np.pi / zeta * F0 
    
    V_prim = N_alpha * N_beta * K_AB * V_unnormalised
    
    return V_prim

def build_S_and_T_matrices(basis_functions):
    """Build the overlap matrix"""
    n_basis = len(basis_functions)
    S = np.zeros((n_basis, n_basis))
    T = np.zeros((n_basis, n_basis))

    for i, basis_i in enumerate(basis_functions):
        for j, basis_j in enumerate(basis_functions):
            R_A = basis_i['center']
            R_B = basis_j['center']

            # Loop over every primitve for the contracted Gaussians
            for alpha, coeff_A in zip(basis_i['exponents'], basis_i['coefficients']):
                for beta, coeff_B in zip(basis_j['exponents'], basis_j['coefficients']):

                    S_prim = compute_S_primitive(alpha, beta, R_A, R_B)
                    T_prim = compute_T_primitive(alpha, beta, R_A, R_B, S_prim)

                    # Add weighted contributions to contracted integrals
                    S[i, j] += coeff_A * coeff_B * S_prim
                    T[i, j] += coeff_A * coeff_B * T_prim
    return S, T

def build_V_nuc_matrix(basis_functions, atoms):
    """
    Build nuclear attraction matrix
    V[i,j] = sum over nuclei of -Z * <chi_i | 1/|r-R_nuc| | chi_j>
    """
    n = len(basis_functions)
    V_nuc = np.zeros((n, n))
    
    for i, basis_i in enumerate(basis_functions):
        for j, basis_j in enumerate(basis_functions):
            R_A = basis_i['center']
            R_B = basis_j['center']
            
            # Sum attraction to each nucleus
            for atom in atoms:
                R_nuc = atom['coords']
                Z = 1.0  # Hydrogen ONLY
                
                # Contract over primitives
                for alpha, coeff_A in zip(basis_i['exponents'], basis_i['coefficients']):
                    for beta, coeff_B in zip(basis_j['exponents'], basis_j['coefficients']):
                        
                        S_prim = compute_S_primitive(alpha, beta, R_A, R_B)
                        V_prim = compute_V_nuc_primitive(alpha, beta, R_A, R_B, R_nuc)
                        
                        V_nuc[i, j] += -Z * coeff_A * coeff_B * V_prim
    
    return V_nuc

def compute_ERI_primitive(alpha, beta, gamma, delta, R_A, R_B, R_C, R_D):
    """
    Compute (ab|cd) electron repulsion integral between four primitive s-type Gaussians
    Uses the Obara-Saika scheme. Entirely generated from Sonnet 4.5 using reference materials
    """
    # Gaussian product theorem for first pair (a,b)
    zeta = alpha + beta
    P = (alpha * R_A + beta * R_B) / zeta
    AB_sq = np.sum((R_A - R_B)**2)
    K_AB = np.exp(-alpha * beta / zeta * AB_sq)

    # Gaussian product theorem for second pair (c,d)
    eta = gamma + delta
    Q = (gamma * R_C + delta * R_D) / eta
    CD_sq = np.sum((R_C - R_D)**2)
    K_CD = np.exp(-gamma * delta / eta * CD_sq)

    # Distance between the two Gaussian products
    PQ_sq = np.sum((P - Q)**2)

    # Compute the Boys function F_0(x) - for s-orbitals we only need the zeroth order
    rho = zeta * eta / (zeta + eta)
    T = rho * PQ_sq

    if T < 1e-10:
        F0 = 1.0  # Limit as T -> 0
    else:
        # F_0(T) = (1/2) * sqrt(π/T) * erf(sqrt(T))
        F0 = 0.5 * np.sqrt(np.pi / T) * math.erf(np.sqrt(T))

    # Normalization constants for s-type Gaussians
    N_alpha = (2.0 * alpha / np.pi)**(0.75)
    N_beta = (2.0 * beta / np.pi)**(0.75)
    N_gamma = (2.0 * gamma / np.pi)**(0.75)
    N_delta = (2.0 * delta / np.pi)**(0.75)

    # Full ERI primitive formula
    ERI_prim = (2.0 * np.pi**2.5 / (zeta * eta * np.sqrt(zeta + eta))) * \
          K_AB * K_CD * F0 * N_alpha * N_beta * N_gamma * N_delta

    return ERI_prim

def build_ERI_tensor(basis_functions):
    """
    Build the 4D tensor of electron repulsion integrals
    ERI[i,j,k,l] = (ij|kl) in chemist's notation
    """
    n_basis = len(basis_functions)
    ERIs = np.zeros((n_basis, n_basis, n_basis, n_basis))

    for i, basis_i in enumerate(basis_functions):
        for j, basis_j in enumerate(basis_functions):
            for k, basis_k in enumerate(basis_functions):
                for l, basis_l in enumerate(basis_functions):
                    R_A = basis_i['center']
                    R_B = basis_j['center']
                    R_C = basis_k['center']
                    R_D = basis_l['center']

                    # Contract over all primitives
                    for alpha, coeff_A in zip(basis_i['exponents'], basis_i['coefficients']):
                        for beta, coeff_B in zip(basis_j['exponents'], basis_j['coefficients']):
                            for gamma, coeff_C in zip(basis_k['exponents'], basis_k['coefficients']):
                                for delta, coeff_D in zip(basis_l['exponents'], basis_l['coefficients']):

                                    ERI_prim = compute_ERI_primitive(
                                        alpha, beta, gamma, delta,
                                        R_A, R_B, R_C, R_D
                                    )

                                    ERIs[i, j, k, l] += coeff_A * coeff_B * coeff_C * coeff_D * ERI_prim

    return ERIs

def build_G_matrix(D, ERIs):
    """
    Build the two-electron part of the Fock matrix: G = J - K
    J is the Coulomb term, K is the exchange term

    G[μ,ν] = Σ_λσ D[λ,σ] * [(μν|λσ) - 0.5*(μλ|νσ)]
    """
    n = D.shape[0]
    G = np.zeros((n, n))

    for mu in range(n):
        for nu in range(n):
            for lam in range(n):
                for sig in range(n):
                    # Coulomb term: (μν|λσ)
                    J = ERIs[mu, nu, lam, sig]
                    # Exchange term: (μλ|νσ)
                    K = ERIs[mu, lam, nu, sig]

                    G[mu, nu] += D[lam, sig] * (J - 0.5 * K)

    return G
