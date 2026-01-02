use std::fs;
use serde_json;
use log::debug;
use ndarray::Array2;
use ndarray_linalg::{Eig, c64};

mod integrals; // bring in integrals from a external module
use integrals::dist_sq;

#[derive(Debug, Clone)]
pub struct BasisSetData {
    pub name: String,
    pub description: String,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
}


// Gemini pointed out I only need to use the normalization of the basis functions once when loading them
impl BasisSetData {
    pub fn normalise(&mut self) {
        let mut total_self_overlap = 0.0;
        let origin = [0.0, 0.0, 0.0];

        // Calculate the current self-overlap of the contracted function
        for (&alpha, &c_i) in self.exponents.iter().zip(self.coefficients.iter()) {
            for (&beta, &c_j) in self.exponents.iter().zip(self.coefficients.iter()) {
                // We use s_primitive at the same center (origin) to find self-norm
                let s_prim = integrals::compute_s_primitive(alpha, beta, &origin, &origin);
                total_self_overlap += c_i * c_j * s_prim;
            }
        }

        // The factor needed to make total_self_overlap == 1.0
        let norm_factor = 1.0 / total_self_overlap.sqrt();

        // DEBUG PRINT
        debug!("DEBUG: Current overlap was {}, applying factor {}", total_self_overlap, norm_factor);

        // Scale all coefficients by this factor
        for c in self.coefficients.iter_mut() {
            *c *= norm_factor;
        }
    }
}

fn load_basis_sets(basis_sets_dir: &str) -> Vec<BasisSetData> {
    let mut basis_sets = Vec::new();

    for entry in fs::read_dir(basis_sets_dir).expect("Failed to read basis set directory") { 
        let entry = entry.expect("Failed to read directory entry");

        if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
            let path = entry.path();
            let json = fs::read_to_string(&path).expect("Basis set unavailable or unreadable");
            
            let value: serde_json::Value = serde_json::from_str(&json).expect("Failed to parse basis set JSON");

            let name = value["name"].as_str().unwrap_or("Unknown").to_string();
            let description = value["description"].as_str().unwrap_or("No description").to_string();

            // Extract electron shell data for Hydrogen (atomic number 1) as an example
            let shell = &value["elements"]["1"]["electron_shells"][0];

            let exponents: Vec<f64> = shell["exponents"]
                .as_array()
                .expect("Exponents should be an array")
                .iter()
                .map(|v| v.as_str().expect("Originally a string").parse::<f64>().expect("String shoulld parse to f64"))
                .collect();

            let coefficients: Vec<f64> = shell["coefficients"][0]
                .as_array()
                .expect("Coefficients should be an array")
                .iter()
                .map(|v| v.as_str().expect("Originally a string").parse::<f64>().expect("String shoulld parse to f64"))
                .collect();

            debug!("Loaded basis set: {} ", description);

            basis_sets.push(BasisSetData {
                name, 
                description,
                exponents,
                coefficients,
            });
        }
    }
    basis_sets
}

fn debug_matrix_values(basis: BasisSetData, basis_functions: Vec<BasisSetData>, r_a: [f64; 3], r_b: [f64; 3]) {

    let s_primitive = integrals::compute_s_primitive(
        basis.exponents[0],
        basis.exponents[0],
        &r_a,
        &r_b
    );
    debug!("Overlap integral (primitive): {}", s_primitive);

    let compute_v_nuc_primitive = integrals::compute_v_nuc_primitive(
        basis.exponents[0],
        basis.exponents[0],
        &r_a,
        &r_b,
        &r_a, // nucleus at atom A
        1.0   // nuclear charge of hydrogen
    );
    debug!("Nuclear attraction integral (primitive): {}", compute_v_nuc_primitive);


    let (s_same, _, _) = integrals::build_one_electron_matrices(&basis_functions, &r_a, &r_a);
    debug!("Self-overlap (same center): {}", s_same[[0,0]]);

    // This is the physical overlap between atom A and atom B
    let (s_diff, _, _) = integrals::build_one_electron_matrices(&basis_functions, &r_a, &r_b);
    debug!("Inter-atomic overlap: {}", s_diff[[0,0]]);

}

fn main() {
    env_logger::init();

    let r_a = [0.0, 0.0, 0.0];
    let r_b = [0.0, 0.0, 1.4]; 
    let r = dist_sq(&r_a, &r_b).sqrt();
    debug!("Inter-nuclear distance: {}", r);

    let basis_sets = load_basis_sets("basis_sets");
    println!("\n=== Loaded {} basis sets ===\n", basis_sets.len());

    let basis_name = "STO-3G";
    let basis = basis_sets
        .iter()
        .find(|bs| bs.name == basis_name)
        .expect("Basis set not found");
    
    debug!("{:#?}", basis);
    debug!("Basis function exponents: {:?}", &basis.exponents);
    debug!("Basis function coefficients: {:?}", &basis.coefficients);
    
    let mut basis = basis.clone(); //needs to be mutable to normalise
    basis.normalise();
    debug!("Post-norm coefficients: {:?}", basis.coefficients);

    let basis_functions = vec![basis.clone(), basis.clone()]; // two basis functions, one on each atom
    let n_basis = basis_functions.len();

    debug_matrix_values(basis, basis_functions.clone(), r_a, r_b);

    let (s_matrix, t_matrix, v_matrix) = integrals::build_one_electron_matrices(&basis_functions, &r_a, &r_b);
    debug!("Overlap matrix S:\n{}", s_matrix);
    debug!("Kinetic energy matrix T:\n{}", t_matrix);
    debug!("Nuclear attraction matrix V:\n{}", v_matrix);
    
    // Build density matrix D with all zeroes as a guess
    let mut d_matrix = Array2::<f64>::zeros((n_basis, n_basis));
    
    let eri_tensor = integrals::build_eri_tensor_symmetric(&basis_functions, &[r_a, r_b]);
    debug!("Electron repulsion integral tensor ERI:\n{:?}", eri_tensor);

    let h_core = &t_matrix + &v_matrix;
    debug!("Core Hamiltonian H_core:\n{}", h_core);

    let e_nuc_rep = 1.0 * 1.0 / r;
    debug!("Nuclear repulsion energy: {}", e_nuc_rep);

    let mut e_old = 0.0;
    let max_iter = 50;
    let conv_thres = 1e-9;

    for iter in 0..max_iter {
        let g_matrix = integrals::build_g_matrix(&eri_tensor, &d_matrix);
        debug!("G matrix:\n{}", g_matrix);
        let f_matrix= &t_matrix + &v_matrix + &g_matrix;
        debug!("Fock matrix F:\n{}", f_matrix);

        let (s_eigvals, s_eigvecs) = s_matrix.eig().expect("Eigendecomposition of S failed");
        debug!("Eigenvalues of S:\n{}", s_eigvals);
        debug!("Eigenvectors of S:\n{}", s_eigvecs);

        // clean up S^-1/2 in case of tiny eigenvalues
        let s_inv_sqrt_diag = Array2::from_diag(
            &s_eigvals.map(|v: &c64| {
                let val = v.re.max(1e-15); 
                val.powf(-0.5)
            })
        );

        // Only the real part needed since S is symmetric real. Julia handled this automatically
        let u = s_eigvecs.map(|v| v.re);
        let x = u.dot(&s_inv_sqrt_diag).dot(&u.t());

        let f_prime = x.t().dot(&f_matrix).dot(&x);
        debug!("Transformed Fock matrix F':\n{}", f_prime);

        // Diagnonalize F' to get orbital energies and coefficients in orthonormal basis    
        let (epsilon_complex, c_prime_complex) = f_prime.eig().expect("Fock diagonalization failed");
        let epsilon = epsilon_complex.map(|v| v.re);
        let c_prime = c_prime_complex.map(|v| v.re);
        debug!("Orbital energies: {:?}", epsilon);

        // back-transform coefficients to original basis
        let c = x.dot(&c_prime);
        debug!("Molecular orbital coefficients:\n{}", c);


        let num_electrons = 2; // Hardcoding for now 
        if num_electrons % 2 != 0 {
            panic!("Restricted Hartree-Fock requires an even number of electrons!");
        }

        // In Rust the eigenvalues from LAPACK are not automatically sorted, so we need to sort them and the corresponding coefficients
        // Unlike Julia, Rust does not have built-in sorting that returns indices, so we create a vector of indices and sort that
        let mut indices: Vec<usize> = (0..epsilon.len()).collect();
        indices.sort_by(|&i, &j| epsilon[i].partial_cmp(&epsilon[j]).unwrap());

        let lowest_idx = indices[0];
        let c_occ = c.column(lowest_idx);

        // Rust magic I needed a lot of LLM help for 
        let c_view = c_occ.view().insert_axis(ndarray::Axis(1));
        d_matrix = 2.0 * c_view.dot(&c_view.t());
        for i in 0..n_basis {
            for j in 0..n_basis {
                d_matrix[[i, j]] = 0.0;
                    d_matrix[[i, j]] += 2.0 * c_occ[[i]] * c_occ[[j]]; 
            }
        }
        debug!("Density matrix D:\n{}", d_matrix);

        let e_elec = 0.5 * (&d_matrix * (&h_core + &f_matrix)).sum();
        debug!("Electronic energy: {}", e_elec);
        debug!("Nuclear repulsion energy: {}", e_nuc_rep);
        let e_total = e_elec + e_nuc_rep;

        let delta_e = (e_total - e_old).abs();
        println!("Energy: {:.10} Hartrees, Delta: {:.10}", e_total, delta_e);

        if delta_e < conv_thres {
            println!("SCF converged in {} iterations.", iter + 1);
            println!("🦀");
            let electrons = (&d_matrix * &s_matrix).sum();
            debug!("Total electrons in system: {:.4}", electrons);
            break;
        }
        e_old = e_total;
    }
}
