use ndarray::Array2;
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

pub mod integrals;
use integrals::{
    BasisSetData, build_one_electron_matrices, build_eri_tensor_symmetric, build_g_matrix
};


// Handle data from the Basis Set Exchange (Bse)
#[derive(Debug, Deserialize)]
struct BseJson {
    pub family: String,
    pub description: String,
    pub elements: HashMap<String, BseElement>,
}

#[derive(Debug, Deserialize)]
struct BseElement {
    pub electron_shells: Vec<BseShell>,
}

#[derive(Debug, Deserialize)]
struct BseShell {
    pub angular_momentum: Vec<usize>,
    pub exponents: Vec<String>,
    pub coefficients: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Primitive {
    pub alpha: f64,
    pub coefficient: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractedShell {
    pub angular_momentum: usize, // 0 = S, 1 = P, 2 = D
    pub center: [f64; 3],        // Physical XYZ coordinates of the atom
    pub primitives: Vec<Primitive>,
}

pub struct MoleculeBasis {
    pub shells: Vec<ContractedShell>,
    pub total_basis_functions: usize,
}

/// Results from an HF calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct HFResult {
    pub converged: bool,
    pub final_energy: f64,
    pub iterations: u32,
    pub nuclear_repulsion: f64,
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
pub fn calculate_scf(bond_dist: f64, basis_name: &str) -> HFResult {
    let hydrogen_shells = get_basis_set(basis_name);

    for (idx, shell) in hydrogen_shells.iter().enumerate() {
       println!(
           "Shell {} -> Exponents: {}, Coefficients: {}", 
           idx, shell.exponents.len(), shell.coefficients.len()
       );
    }
 
    // H2 geometry: atom A at origin, atom B at (0, 0, bond_dist)
    let r_a = [0.0, 0.0, 0.0];
    let r_b = [0.0, 0.0, bond_dist];
   
    let mut basis_functions = Vec::new();
 
   // First half: Clone the entire suite of shells for Atom A
   basis_functions.extend(hydrogen_shells.iter().cloned());
   // Second half: Clone the entire suite of shells for Atom B
   basis_functions.extend(hydrogen_shells.iter().cloned());   
    
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
        
        // Use nalgebra for eigendecomposition (works on WASM + native)
        use nalgebra as na;
        
        let s_na = na::DMatrix::<f64>::from_row_slice(n_basis, n_basis, s_matrix.as_slice().unwrap());
        let eig_s = s_na.symmetric_eigen();
        
        // Create S^(-1/2)
        let s_inv_sqrt_diag = na::DMatrix::from_diagonal(
            &eig_s.eigenvalues.map(|v| v.max(1e-15).powf(-0.5))
        );
        
        let x = &eig_s.eigenvectors * &s_inv_sqrt_diag * eig_s.eigenvectors.transpose();
        
        // Transform Fock matrix
        let f_na = na::DMatrix::<f64>::from_row_slice(n_basis, n_basis, f_matrix.as_slice().unwrap());
        let f_prime = &x.transpose() * &f_na * x.clone(); // clone X matrix here so it remains available for recomputin coeff matrix
        let eig_f = f_prime.symmetric_eigen();       
 
        let epsilon = eig_f.eigenvalues;
        let c_prime = eig_f.eigenvectors;
        let c = &x * &c_prime;
        
        // Convert back to ndarray - fix Column-major into Row-major nalgebra ndarray bug
        let c_row_major = c.transpose();
        let c = Array2::from_shape_vec(
            (n_basis, n_basis),
            c_row_major.as_slice().to_vec()
        ).unwrap();        
 
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

/// Embedded basis sets for WASM with strings generated a compile time (no file I/O)
fn get_basis_set(name: &str) -> Vec<BasisSetData> {
    let raw_json = match name {
     // "STO-2G" => include_str!("../basis_sets/sto-2g-H.json"),
     // "STO-3G" => include_str!("../basis_sets/sto-3g-H.json"),
     // "STO-4G" => include_str!("../basis_sets/sto-4g-H.json"),
     // "STO-5G" => include_str!("../basis_sets/sto-5g-H.json"),
     // "STO-6G" => include_str!("../basis_sets/sto-6g-H.json"),
        "MINI"   => include_str!("../basis_sets/MINI-H.json"),
        "3-21G"  => include_str!("../basis_sets/3-21g-H.json"),
        "6-31G"  => include_str!("../basis_sets/6-31g-H.json"),
        "6-311G" => include_str!("../basis_sets/6-311g-H.json"),
        "cc-pVDZ"  => include_str!("../basis_sets/cc-pvDZ-H.json"),
        "cc-pVTZ"  => include_str!("../basis_sets/cc-pvTZ-H.json"),
        "cc-pVQZ"  => include_str!("../basis_sets/cc-pvQZ-H.json"),
        _ => panic!("Unsupported basis set name: {}", name), 
      };

      let parsed: BseJson = serde_json::from_str(raw_json)
        .unwrap_or_else(|_| panic!("Failed to parse embedded JSON for {}", name));

      let is_dunning = parsed.family.trim() == "dunning";
      
      // Only apply the Pople molecular orbital scaling to raw minimal fits.
      // 3-21G, 6-31G, and Dunning sets are already pre-scaled.
      let zeta_squared = match parsed.family.trim() {
           "sto" => 1.24 * 1.24, // Minimal basis set scaling
           "huzinaga"   => 1.25 * 1.25, // Huzinaga MINI molecular scaling
           _        => 1.0, // Baseline fallback
       };
      
      let mut shells = Vec::new();
      let element_data = parsed.elements.get("1").expect("Missing Hydrogen (element 1) in the basis set json");
      
      for shell_data in &element_data.electron_shells {
        if shell_data.angular_momentum != vec![0] {
            continue;
        }
   
        let raw_exponents: Vec<f64> = shell_data.exponents
            .iter()
            .map(|exp_str| exp_str.parse::<f64>().unwrap())
            .collect();

        // If it's Dunning, take all rows. Otherwise, just take the first row.
        let limit = if is_dunning { shell_data.coefficients.len() } else { 1 };

        for coef_row in shell_data.coefficients.iter().take(limit) {
            let coefficients: Vec<f64> = coef_row
                .iter()
                .map(|coef_str| coef_str.parse::<f64>().unwrap())
                .collect();

            // 1. Scale exponents first to establish the final molecular coordinate field
            let mut scaled_exponents = raw_exponents.clone();
            if zeta_squared != 1.0 {
                for alpha in &mut scaled_exponents {
                    *alpha *= zeta_squared;
                }
            }

            // 2. Build the function tracking the scaled exponents directly
            let mut basis_fn = BasisSetData {
                name: name.to_string(),
                description: parsed.description.clone(),
                exponents: scaled_exponents,
                coefficients,
            };
              
            // 3. Normalize the contraction coefficients inside the final scaled framework
            basis_fn.normalise();

            shells.push(basis_fn);
        }
    }

    shells
}

// Expose integrals module for library use
pub use integrals::{compute_s_primitive, compute_t_primitive, compute_v_nuc_primitive};

