use std::f64::consts::PI;
//use ndarray::Array2;

pub fn compute_S_primitive (alpha: f64, beta: f64, r_a: &[f64; 3], r_b: &[f64; 3]) -> f64 {
    // Normalisation factor for primitive s-type Gaussian functions
    let norm_alpha = (2.0 * alpha / PI).powf(0.75);
    let norm_beta = (2.0 * beta / PI).powf(0.75);

    let distance_squared: f64 = (r_a[0] - r_b[0]).powi(2)
        + (r_a[1] - r_b[1]).powi(2)
        + (r_a[2] - r_b[2]).powi(2);

    let exponent =  (alpha * beta / (alpha + beta)) * distance_squared;
    let s_unnormalized = (PI / (alpha + beta)).powf(1.5) * exponent.exp();

    norm_alpha * norm_beta * s_unnormalized
}


// pub fn build_S_and_T_matrices (basis_functions: &[BasisFunctions]) -> (Array2<f64>, Array2<f64>) {
//     let n_basis = basis_functions.len();
//     let mut S_matrix = Array2::<f64>::zeros((n_basis, n_basis));
//     let mut T_matrix = Array2::<f64>::zeros((n_basis, n_basis));

//     for i in 0..n_basis {
//         let basis_functions_i = &basis_functions[i];
//         for j in 0..n_basis {
//             let basis_functions_j = &basis_functions[j];
//             r_a = &basis_functions_i.center;
//             r_b = &basis_functions_j.center;

//             // Create contracted Gaussians by looping over primitives
//             for (alpha, coeff_alpha) in basis_functions_i.primitives.iter().zip(basis_functions_i.coefficients.iter()) {
//                 for (beta, coeff_beta) in basis_functions_j.primitives.iter().zip(basis_functions_j.coefficients.iter()) {

//                     let s_primitive = compute_S_primitive(*alpha, *beta, r_a, r_b);
//                     S_matrix[[i, j]] += coeff_alpha * coeff_beta * S_primitive;

//                     let T_primitive = compute_T_primitive(*alpha, *beta, r_a, r_b);
//                     T_matrix[[i, j]] += coeff_alpha * coeff_beta * T_primitive;
//                 }
//             }
//         }
//     }
//     (S_matrix, T_matrix) 
// }