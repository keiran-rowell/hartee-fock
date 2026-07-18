use log::debug;
use ndarray::Array2;

mod integrals; // bring in integrals from a external module
use integrals::{BasisSetData, load_basis_sets, dist_sq};

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


    let (s_same, _, _) = integrals::build_one_electron_matrices(basis_functions.as_slice(), &r_a, &r_a);
    debug!("Self-overlap (same center): {}", s_same[[0,0]]);

    // This is the physical overlap between atom A and atom B
    let (s_diff, _, _) = integrals::build_one_electron_matrices(basis_functions.as_slice(), &r_a, &r_b);
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

    let (s_matrix, t_matrix, v_matrix) = integrals::build_one_electron_matrices(basis_functions.as_slice(), &r_a, &r_b);
    debug!("Overlap matrix S:\n{}", s_matrix);
    debug!("Kinetic energy matrix T:\n{}", t_matrix);
    debug!("Nuclear attraction matrix V:\n{}", v_matrix);

    // Build density matrix D with all zeroes as a guess
    let mut d_matrix = Array2::<f64>::zeros((n_basis, n_basis));

    let eri_tensor = integrals::build_eri_tensor_symmetric(basis_functions.as_slice(), &[r_a, r_b]);
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
        let f_matrix = &t_matrix + &v_matrix + &g_matrix;
        debug!("Fock matrix F:\n{}", f_matrix);

        // Use nalgebra for eigendecomposition
        use nalgebra as na;
        
        let s_na = na::Matrix2::<f64>::from_row_slice(s_matrix.as_slice().unwrap());
        let eig_s = s_na.symmetric_eigen();
        
        debug!("Eigenvalues of S: {:?}", eig_s.eigenvalues.as_slice());
        debug!("Eigenvectors of S:\n{}", eig_s.eigenvectors);

        // Create S^(-1/2)
        let s_inv_sqrt_diag = na::Matrix2::from_diagonal(
            &eig_s.eigenvalues.map(|v| v.max(1e-15).powf(-0.5))
        );
        
        let x = &eig_s.eigenvectors * &s_inv_sqrt_diag * eig_s.eigenvectors.transpose();
        
        // Transform Fock matrix
        let f_prime = x.transpose() * na::Matrix2::<f64>::from_row_slice(f_matrix.as_slice().unwrap()) * x;
        debug!("Transformed Fock matrix F':\n{}", f_prime);

        let eig_f = f_prime.symmetric_eigen();
        let epsilon = eig_f.eigenvalues;
        let c_prime = eig_f.eigenvectors;
        let c = x * c_prime;
        
        debug!("Orbital energies: {:?}", epsilon.as_slice());

        // Convert back to ndarray - fix Column-major into Row-major nalgebra ndarray bug
        let c_row_major = c.transpose();
        let c = Array2::from_shape_vec(
            (2, 2),
            c_row_major.as_slice().to_vec()
        ).unwrap();
        debug!("Molecular orbital coefficients:\n{}", c);

        let num_electrons = 2; // Hardcoding for now
        if num_electrons % 2 != 0 {
            panic!("Restricted Hartree-Fock requires an even number of electrons!");
        }

        // Sort eigenvalues and get lowest orbital
        let mut indices: Vec<usize> = (0..epsilon.len()).collect();
        indices.sort_by(|&i, &j| epsilon[i].partial_cmp(&epsilon[j]).unwrap());

        let lowest_idx = indices[0];
        let c_occ = c.column(lowest_idx);

        // Build density matrix
        let mut d_new = Array2::<f64>::zeros((n_basis, n_basis));
        for i in 0..n_basis {
            for j in 0..n_basis {
                d_new[[i, j]] = 2.0 * c_occ[[i]] * c_occ[[j]];
            }
        }
        d_matrix = d_new;
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

