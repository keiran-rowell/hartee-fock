use std::f64::consts::PI;
use ndarray::Array2;

use crate::BasisSetData;

pub fn compute_s_primitive (alpha: f64, beta: f64, r_a: &[f64; 3], r_b: &[f64; 3]) -> f64 {
    // Normalisation factor for primitive s-type Gaussian functions
    let norm_alpha = (2.0 * alpha / PI).powf(0.75);
    let norm_beta = (2.0 * beta / PI).powf(0.75);

    let distance_squared: f64 = (r_a[0] - r_b[0]).powi(2)
        + (r_a[1] - r_b[1]).powi(2)
        + (r_a[2] - r_b[2]).powi(2);

    let exponent =  -(alpha * beta / (alpha + beta)) * distance_squared;
    let s_unnormalized = (PI / (alpha + beta)).powf(1.5) * exponent.exp();

    norm_alpha * norm_beta * s_unnormalized
}

pub fn compute_t_primitive (alpha: f64,  beta: f64, r_a: &[f64; 3], r_b: &[f64; 3], s_prim: f64) -> f64 {
    let reduced_exponent = (alpha * beta) / (alpha + beta);
    
    let distance_squared: f64 = (r_a[0] - r_b[0]).powi(2)
        + (r_a[1] - r_b[1]).powi(2)
        + (r_a[2] - r_b[2]).powi(2);

    reduced_exponent * (3.0 - 2.0 * reduced_exponent * distance_squared) * s_prim
}

pub fn build_s_and_t_matrices (basis_functions: &[BasisSetData], r_a: &[f64; 3], r_b: &[f64; 3]) -> (Array2<f64>, Array2<f64>) {
    let n_basis = basis_functions.len();
    let mut s_matrix = Array2::<f64>::zeros((n_basis, n_basis));
    let mut t_matrix = Array2::<f64>::zeros((n_basis, n_basis));

    // Basis function i is on atom A, basis function j is on atom B
    for (i, bf_i) in basis_functions.iter().enumerate() {
        for (j, bf_j) in basis_functions.iter().enumerate() { 

        let mut s_val = 0.0;
        let mut t_val = 0.0;

        for (&alpha, &coeff_alpha) in bf_i.exponents.iter().zip(bf_i.coefficients.iter()) {
            for (&beta, &coeff_beta) in bf_j.exponents.iter().zip(bf_j.coefficients.iter()) {

                    let s_prim = compute_s_primitive(alpha, beta, r_a, r_b);
                    let t_prim = compute_t_primitive(alpha, beta, r_a, r_b, s_prim);

                    //contract the primitive integrals with the coefficients
                    s_val += coeff_alpha * coeff_beta * s_prim;
                    t_val += coeff_alpha * coeff_beta * t_prim;
                }
            }
            s_matrix[[i, j]] = s_val;
            t_matrix[[i, j]] = t_val;
        }
    }
    (s_matrix, t_matrix) 
}