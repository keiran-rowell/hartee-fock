use ndarray::Array2;
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use ndarray_linalg::Eig;

pub mod integrals;
use integrals::{
    BasisSetData, build_one_electron_matrices, build_eri_tensor_symmetric, build_g_matrix
};



/// Results from an HF calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct HFResult {
    pub converged: bool,
    pub final_energy: f64,
    pub iterations: u32,
    pub nuclear_repulsion: f64,
}

#[wasm_bindgen]
impl HFResult {
    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool {
        self.converged
    }

    #[wasm_bindgen(getter)]
    pub fn final_energy(&self) -> f64 {
        self.final_energy
    }

    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    #[wasm_bindgen(getter)]
    pub fn nuclear_repulsion(&self) -> f64 {
        self.nuclear_repulsion
    }
}

/// Run HF calculation for H2 at given bond distance
/// bond_dist: internuclear distance in Bohr
/// Returns: HFResult with energy and convergence info
#[wasm_bindgen]
pub fn run_hf_wasm(bond_dist: f64) -> HFResult {
    calculate_scf(bond_dist, "STO-3G")
}

/// Run HF calculation for H2 with specified basis set
/// bond_dist: internuclear distance in Bohr
/// basis_name: name of basis set (e.g. "STO-3G")
#[wasm_bindgen]
pub fn run_hf_with_basis(bond_dist: f64, basis_name: &str) -> HFResult {
    calculate_scf(bond_dist, basis_name)
}

/// Core SCF calculation
fn calculate_scf(bond_dist: f64, basis_name: &str) -> HFResult {
    // Hardcoded basis sets for WASM (embedded)
    let basis = get_basis_set(basis_name);
    
    // H2 geometry: atom A at origin, atom B at (0, 0, bond_dist)
    let r_a = [0.0, 0.0, 0.0];
    let r_b = [0.0, 0.0, bond_dist];
    
    // Create basis functions: one on each atom
    let mut basis_a = basis.clone();
    let mut basis_b = basis.clone();
    
    basis_a.normalise();
    basis_b.normalise();
    
    let basis_functions = vec![basis_a, basis_b];
    let n_basis = basis_functions.len();
    
    // Build one-electron matrices
    let (s_matrix, t_matrix, v_matrix) = build_one_electron_matrices(
        &basis_functions,
        &r_a,
        &r_b,
    );
    
    // Build electron repulsion integral tensor
    let eri_tensor = build_eri_tensor_symmetric(&basis_functions, &[r_a, r_b]);
    
    // Core Hamiltonian
    let h_core = &t_matrix + &v_matrix;
    
    // Nuclear repulsion energy (Born-Oppenheimer)
    let e_nuc_repulsion = 1.0 * 1.0 / bond_dist;
    
    // SCF loop
    let mut d_matrix = Array2::<f64>::zeros((n_basis, n_basis));
    let mut e_old = 0.0;
    let max_iter = 50;
    let conv_thresh = 1e-9;
    let mut converged = false;
    let mut iteration = 0;
    let mut e_total = 0.0;
    
    for iter in 0..max_iter {
        iteration = iter + 1;
        
        // Build Fock matrix
        let g_matrix = build_g_matrix(&eri_tensor, &d_matrix);
        let f_matrix = &t_matrix + &v_matrix + &g_matrix;
        
        // Orthogonalization: X = U * S^(-1/2) * U^T
        let (s_eigvals, s_eigvecs) = s_matrix.eig().expect("Eigendecomposition of S failed");
        
        // Clean up eigenvalues to avoid numerical issues
        let s_inv_sqrt_diag = ndarray::Array2::from_diag(
            &s_eigvals.map(|v| {
                let val = v.re.max(1e-15);
                val.powf(-0.5)
            }),
        );
        
        let u = s_eigvecs.map(|v| v.re);
        let x = u.dot(&s_inv_sqrt_diag).dot(&u.t());
        
        // Transform Fock matrix to orthonormal basis
        let f_prime = x.t().dot(&f_matrix).dot(&x);
        
        // Diagonalize transformed Fock matrix
        let (epsilon_complex, c_prime_complex) = f_prime.eig().expect("Fock diagonalization failed");
        let epsilon = epsilon_complex.map(|v| v.re);
        let c_prime = c_prime_complex.map(|v| v.re);
        
        // Back-transform coefficients to original basis
        let c = x.dot(&c_prime);
        
        // Build density matrix from occupied orbitals
        let num_electrons = 2; // H2 has 2 electrons
        if num_electrons % 2 != 0 {
            panic!("Restricted HF requires even number of electrons");
        }
        
        // Sort eigenvalues and get indices of lowest orbitals
        let mut indices: Vec<usize> = (0..epsilon.len()).collect();
        indices.sort_by(|&i, &j| epsilon[i].partial_cmp(&epsilon[j]).unwrap());
        
        // Build density matrix from lowest occupied orbital
        let mut d_new = Array2::<f64>::zeros((n_basis, n_basis));
        for i in 0..n_basis {
            for j in 0..n_basis {
                d_new[[i, j]] = 2.0 * c[[i, indices[0]]] * c[[j, indices[0]]];
            }
        }
        
        // Calculate energy
        let e_elec = 0.5 * (&d_new * (&h_core + &f_matrix)).sum();
        e_total = e_elec + e_nuc_repulsion;
        
        let delta_e = (e_total - e_old).abs();
        
        // Check convergence
        if delta_e < conv_thresh {
            converged = true;
            break;
        }
        
        e_old = e_total;
        d_matrix = d_new;
    }
    
    HFResult {
        converged,
        final_energy: e_total,
        iterations: iteration as u32,
        nuclear_repulsion: e_nuc_repulsion,
    }
}

/// Embedded basis sets for WASM (no file I/O)
fn get_basis_set(name: &str) -> BasisSetData {
    match name {
        "STO-3G" => BasisSetData {
            name: "STO-3G".to_string(),
            description: "STO-3G minimal basis".to_string(),
            exponents: vec![0.168855, 0.623913, 3.425251],
            coefficients: vec![0.444635, 0.535328, 0.154329],
        },
        "STO-2G" => BasisSetData {
            name: "STO-2G".to_string(),
            description: "STO-2G (minimal)".to_string(),
            exponents: vec![0.270950, 1.409570],
            coefficients: vec![0.430129, 0.678914],
        },
        "3-21G" => BasisSetData {
            name: "3-21G".to_string(),
            description: "3-21G split valence".to_string(),
            // Inner shell
            exponents: vec![5.447178, 0.824547, 0.183192],
            coefficients: vec![0.156285, 0.904691, 0.0],
        },
        "6-31G" => BasisSetData {
            name: "6-31G".to_string(),
            description: "6-31G Pople basis".to_string(),
            exponents: vec![18.731137, 2.825394, 0.640122],
            coefficients: vec![0.033495, 0.234727, 0.813757],
        },
        _ => {
            // Default to STO-3G
            BasisSetData {
                name: "STO-3G".to_string(),
                description: "STO-3G minimal basis".to_string(),
                exponents: vec![0.168855, 0.623913, 3.425251],
                coefficients: vec![0.444635, 0.535328, 0.154329],
            }
        }
    }
}

// Expose integrals module for library use
pub use integrals::{compute_s_primitive, compute_t_primitive, compute_v_nuc_primitive};

