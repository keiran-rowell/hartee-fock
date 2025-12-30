"""
Toy Hartree-Fock implementation for H2
Demonstrates SCF convergence without full ERI implementation
Uses density damping to create iterative behavior
Intended to show the main components of the SCF cycles without getting lost in the maths
"""

import basis_set_exchange
import numpy as np
import json
import logging

# Set up logging so intermediate matrices only print in debug mode (logging.DEBUG)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Construct the molecule
### could use 'from pyscf import gto' or OpenBabel or other options
def read_xyz(file_path):
    """Read an XYZ file and return atomic coordinates."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    atom_lines = lines[2:]  # Skip the first two lines (header)
    atoms = []
    for line in atom_lines:
        parts = line.split()
        atom = {
            'element': parts[0],
            'coords': np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        }
        atoms.append(atom)
    return atoms

## Read xyz file to set atomic positions
atoms = read_xyz('../h2.xyz')
R_A = atoms[0]['coords']
R_B = atoms[1]['coords']
R = np.linalg.norm(R_A - R_B) # Simple internuclear distance
Z = 1.0 if atoms[0]['element'] == 'H' else None # Not implementing other atoms for toy example

# Parse basis sets
basis_set_json =  basis_set_exchange.get_basis('STO-3G', elements=['1'], fmt='json', header=False)
basis_set_data = json.loads(basis_set_json)
logger.debug(f'basis_set_data:\n {basis_set_data}')
## Basis sets in searchable JSON form!

## Count number of basis functions
n_shells = basis_set_data['elements']['1']['electron_shells']
angular_momenta = [l for shell in n_shells for l in shell['angular_momentum']]
n_functions_per_atom = sum([2*l + 1 for l in angular_momenta])
n_basis = 2 * n_functions_per_atom
logger.debug(f'n_basis: {n_basis}')

## Get a list of exponents (alpha values for the primitives) and contraction coefficients (d values)
basis_functions = []

for shell in n_shells:
    exponents = np.array([float(exp) for exp in shell['exponents']])

    for contraction in shell['coefficients']:
        contraction_coeffs = np.array([float(coeff) for coeff in contraction])

        # Store the basis function data per atom
        for atom in atoms:
            basis_functions.append({
                'center': atom['coords'],
                'exponents': exponents,
                'coefficients': contraction_coeffs,
            })

logger.debug(f'exponents: {exponents}')
logger.debug(f'contraction_coeffs: {contraction_coeffs}')

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

# Set up operator matrices

## Intitialise Overlap and Kinetic Energy Matrix
S = np.zeros((n_basis, n_basis))
T = np.zeros((n_basis, n_basis))

# Initialised Density Matrix with random pertubations instead of zeroes to avoid rapid convergence
D = np.random.rand(n_basis, n_basis) * 0.01
# Symmetrised Density Matrix
D = (D + D.T) / 2

## Nuclear attraction Potential Matrix (ultra simplified dummy, no integrals implemented)
V_nuc = -Z * np.ones((n_basis, n_basis)) / R

# Populate the Overlap and Kinetic Energy Matrices
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

logger.info(f'S overlap matrix:\n {S}')
logger.info(f'T kinetic energy matrix:\n {T}')

# Set up the SCF loop
max_iter = 50
epsilon_tol = 1e-6
converged = False
E_old = 0.0

logger.info('\n🔁 Starting SCF iterations...')

for iteration in range(max_iter):
    ## Build the Fock Matrix 
    ### Ignoring ERIs so a fake dampening factor on the electron density D involved)
    density_dampening = 0.9 # Increase to have the density have more effect on the self-consistent field, generally increases num_iter
    F = T + V_nuc + density_dampening * D

    ## Calculate Orthogonalisation Matrix
    s_eigvals, s_eigvecs = np.linalg.eigh(S)
    X = s_eigvecs @ np.diag(s_eigvals**-0.5) @ s_eigvecs.T
    logger.debug(f'X orthogonalisation matrix:\n {X}')

    ## Transform Fock Matrix into new basis
    F_prime = X.T @ F @ X
    logger.debug(f'F\' transformed Fock matrix:\n {F_prime}')

    # Update Coefficient Matrix
    epsilon, C_prime = np.linalg.eigh(F_prime)
    logger.info(f'epsilon orbital energies:\n {epsilon}')

    C = X @ C_prime
    logger.info(f'C coefficient matrix:\n {C}')

    # Update Density Matrix
    num_electrons = len(atoms) # Assuming each H contributes 1 electron. Not implementing proper count
    if num_electrons % 2 != 0:
        logger.error("Number of electrons is odd, cannot proceed with restricted HF.")
    num_occ = num_electrons // 2
    
    D_new = 2 * C[:, :int(num_occ)] @ C[:, :int(num_occ)].T
    logger.debug(f'D_new density matrix:\n {D_new}')

    E_new = 0.5 * np.trace(D_new @ (T + V_nuc + F))
    delta_E = abs(E_new - E_old)

    logger.info(f'\nIteration {iteration + 1}: E = {E_new:.6f} Ha, ΔE = {delta_E:.2e}\n')

    # Check for SCF convergence
    if delta_E < epsilon_tol:
        logger.info(f"\n✓ SCF Converged in {iteration+1} iterations!")
        converged = True
        break
    else:
        E_old = E_new
        D = D_new

if not converged:
    logger.info("\n✗ SCF did not converge within the maximum number of iterations.")

logger.info(f'Final SCF Energy: {E_new:.6f} Ha')
logger.info(f'Final orbital energies: {epsilon}')
