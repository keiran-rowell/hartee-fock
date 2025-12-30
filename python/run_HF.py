import basis_set_exchange
import numpy as np
import json
import logging
import time
import io
import sys

from integrals import (
    build_S_and_T_matrices,
    build_V_nuc_matrix,
    build_ERI_tensor,
    build_G_matrix,
)


def main(R=1.4, basis_set_name="STO-3G"):
    """
    Toy Hartree-Fock implementation for H2
    Demonstrates SCF convergence
    Intended to show the main components of the SCF cycles without getting lost in the maths
    """

    # Set up logging so intermediate matrices only print in debug mode (logging.DEBUG)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    # Add a handler to capture output
    log_capture = io.StringIO()
    capture_handler = logging.StreamHandler(log_capture)
    capture_handler.setLevel(logging.INFO)
    logger.addHandler(capture_handler)

    # Hardcode H2 for this web demo since that's all the slider opens 
    atoms = [
        {'element': 'H', 'coords': np.array([0.0, 0.0, 0.0])},
        {'element': 'H', 'coords': np.array([0.0, 0.0, R])}
    ]
    Z = 1.0

    # Parse basis sets
    basis_set_json = basis_set_exchange.get_basis(
        basis_set_name, elements=["1"], fmt="json", header=False
    )
    basis_set_data = json.loads(basis_set_json)
    logger.debug(f"basis_set_data:\n {basis_set_data}")
    ## Basis sets in searchable JSON form!

    ## Get a list of exponents (alpha values for the primitives) and contraction coefficients (d values)
    basis_functions = []
    n_shells = basis_set_data["elements"]["1"]["electron_shells"]
    for shell in n_shells:
        exponents = np.array([float(exp) for exp in shell["exponents"]])

        for contraction in shell["coefficients"]:
            contraction_coeffs = np.array([float(coeff) for coeff in contraction])

            # Store the basis function data per atom
            for atom in atoms:
                basis_functions.append(
                    {
                        "center": atom["coords"],
                        "exponents": exponents,
                        "coefficients": contraction_coeffs,
                    }
                )

    n_basis = len(basis_functions)
    logger.debug(f"exponents: {exponents}")
    logger.debug(f"contraction_coeffs: {contraction_coeffs}")

    ## Initialised Density Matrix with random pertubations instead of zeroes to avoid rapid convergence
    # D = np.random.rand(n_basis, n_basis) * 0.01
    # Symmetrised Density Matrix
    # D = (D + D.T) / 2

    ## Intialise Density Matrix
    D = np.zeros((n_basis, n_basis))

    ## Nuclear attraction Potential Matrix (ultra simplified dummy, no integrals implemented)
    V_nuc = build_V_nuc_matrix(basis_functions, atoms)
    logger.info(f"V_nuc nuclear attraction matrix:\n {V_nuc}")

    # Populate the one-electron Overlap and Kinetic Energy Matrices (refer to subfunctions in integrals.py)
    S, T = build_S_and_T_matrices(basis_functions)
    logger.info(f"S overlap matrix:\n {S}")
    logger.info(f"T kinetic energy matrix:\n {T}")

    ## Build the ERIs once before SCF loop
    logger.info("Computing electron repulsion integrals...")
    eri_start = time.time()
    ERIs = build_ERI_tensor(basis_functions)
    eri_end = time.time()
    logger.info(f"ERIs computed in {eri_end - eri_start:.3f} seconds")
    logger.debug(f"ERI[0,0,0,0] = {ERIs[0,0,0,0]:.6f}")  # Should be ~0.77 for H2/STO-3G
    logger.debug(f"ERI[0,0,1,1] = {ERIs[0,0,1,1]:.6f}")  # Should be ~0.44

    H_core = T + V_nuc
    logger.info(f"H_core matrix:\n{H_core}")

    # Diagonalize H_core directly
    s_eigvals, s_eigvecs = np.linalg.eigh(S)
    X = s_eigvecs @ np.diag(s_eigvals ** (-0.5)) @ s_eigvecs.T
    H_core_prime = X.T @ H_core @ X
    epsilon_core, C_core_prime = np.linalg.eigh(H_core_prime)
    C_core = X @ C_core_prime

    logger.debug(f"H_core orbital energies: {epsilon_core}")
    logger.debug(f"H_core orbital coefficients:\n{C_core}")

    # Compute nuclear repulsion energy in fixed nuclei approximation (Born-Oppenheimer)
    E_nuc_repulsion = Z * Z / R
    logger.info(f"Nuclear repulsion energy: {E_nuc_repulsion:.6f} Ha")

    # Set up the SCF loop
    max_iter = 50
    epsilon_tol = 1e-6
    converged = False
    E_old = 0.0
    energy_history = [] 

    logger.info("\n🔁 Starting SCF iterations...")
    scf_start = time.time()
    for iteration in range(max_iter):

        ## Build the Fock Matrix
        G = build_G_matrix(D, ERIs)
        logger.debug(f"G matrix:\n{G}")
        F = T + V_nuc + G
        logger.debug(f"H_core:\n{T + V_nuc}")

        ## Calculate Orthogonalisation Matrix
        s_eigvals, s_eigvecs = np.linalg.eigh(S)
        logger.debug(f"S matrix eigenvalues: {s_eigvals}")
        X = s_eigvecs @ np.diag(s_eigvals**-0.5) @ s_eigvecs.T
        logger.debug(f"X orthogonalisation matrix:\n {X}")

        ## Transform Fock Matrix into new basis
        F_prime = X.T @ F @ X
        logger.debug(f"F' transformed Fock matrix:\n {F_prime}")

        # Update Coefficient Matrix
        epsilon, C_prime = np.linalg.eigh(F_prime)
        logger.debug(f"C_prime before transform:\n{C_prime}")
        logger.info(f"epsilon orbital energies:\n {epsilon}")

        C = X @ C_prime
        logger.debug(f"C coefficient matrix:\n {C}")

        # Update Density Matrix
        num_electrons = len(
            atoms
        )  # Assuming each H contributes 1 electron. Not implementing proper count
        if num_electrons % 2 != 0:
            logger.error(
                "Number of electrons is odd, cannot proceed with restricted HF."
            )
        num_occ = num_electrons // 2

        D_new = 2 * C[:, :num_occ] @ C[:, :num_occ].T
        logger.debug(f"D_new density matrix:\n {D_new}")

        E_elec = 0.5 * np.trace(D_new @ (T + V_nuc + F))
        E_total = E_elec + E_nuc_repulsion
        energy_history.append(E_total)
        delta_E = abs(E_total - E_old)

        logger.info(
            f"\nIteration {iteration + 1}: E = {E_total:.6f} Ha, ΔE = {delta_E:.2e}\n"
        )

        # Check for SCF convergence
        if delta_E < epsilon_tol:
            scf_end = time.time()
            logger.info(
                f"\n✓ SCF Converged in {iteration+1} iterations! \n Time taken: {scf_end - scf_start:.3f} seconds"
            )
            converged = True
            break
        else:
            E_old = E_total
            D = D_new

    if not converged:
        logger.info("\n✗ SCF did not converge within the maximum number of iterations.")

    logger.info(f"Final SCF Energy: {E_total:.6f} Ha")
    logger.info(f"Final orbital energies: {epsilon}")

    # Capture all output into a log that the web interface can parse
    log_output = log_capture.getvalue()
    logger.removeHandler(capture_handler)

    return {
        "converged": converged,
        "energy": E_total,
        "iterations": iteration + 1,
        "epsilon": epsilon.tolist(),
        "log": log_output,
        "S_matrix" : S.tolist(),
        "energy_history": energy_history,
    }
