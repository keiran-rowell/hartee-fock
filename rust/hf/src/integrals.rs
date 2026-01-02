use std::f64::consts::PI;
use ndarray::Array2;

use crate::BasisSetData;

pub fn compute_s_primitive (alpha: f64, beta: f64, r_a: &[f64; 3], r_b: &[f64; 3]) -> f64 {
    // Normalisation factor for primitive s-type Gaussian functions
    let norm = (2.0 * alpha / PI).powf(0.75) * (2.0 * beta / PI).powf(0.75);

    let dist_sq: f64 = (r_a[0] - r_b[0]).powi(2)
        + (r_a[1] - r_b[1]).powi(2)
        + (r_a[2] - r_b[2]).powi(2);

    let prefactor = (PI / (alpha + beta)).powf(1.5);
    let exponent =  -(alpha * beta / (alpha + beta)) * dist_sq;

    norm * prefactor * exponent.exp()
}

pub fn compute_t_primitive (alpha: f64,  beta: f64, r_a: &[f64; 3], r_b: &[f64; 3], s_prim: f64) -> f64 {
    let reduced_exponent = (alpha * beta) / (alpha + beta);
    
    let dist_sq: f64 = (r_a[0] - r_b[0]).powi(2)
        + (r_a[1] - r_b[1]).powi(2)
        + (r_a[2] - r_b[2]).powi(2);

    reduced_exponent * (3.0 - 2.0 * reduced_exponent * dist_sq) * s_prim
}

pub fn compute_v_nuc_primitive (alpha: f64, beta: f64, r_a: &[f64; 3], r_b: &[f64; 3], r_nuc: &[f64; 3], z_nuc: f64) -> f64 {
    let norm = (2.0 * alpha / PI).powf(0.75) * (2.0 * beta / PI).powf(0.75);

    let zeta = alpha + beta;
    // Compute the weighted product center of the two Gaussians
    let r_p = [
        (alpha * r_a[0] + beta * r_b[0]) / zeta,
        (alpha * r_a[1] + beta * r_b[1]) / zeta,
        (alpha * r_a[2] + beta * r_b[2]) / zeta,
    ];

   let ab_sq = (r_a[0] - r_b[0]).powi(2) + (r_a[1] - r_b[1]).powi(2) + (r_a[2] - r_b[2]).powi(2);
   let k_ab = (-(alpha * beta / zeta * ab_sq)).exp();

   // Distance from the product center to the nucleus
    let rp_nuc_sq = (r_p[0] - r_nuc[0]).powi(2) + (r_p[1] - r_nuc[1]).powi(2) + (r_p[2] - r_nuc[2]).powi(2);

    //  The Boys function F_0(t) for t = zeta * rp_nuc_sq
    let x = zeta * rp_nuc_sq;
    let f_0 = if x.abs() < 1e-10 {
        1.0
    } else {
        (0.5 * PI.sqrt() / x.sqrt()) * libm::erf(x.sqrt()) 
    };   

    let v_unnrom = (2.0 * PI / zeta) * k_ab * f_0;

    -z_nuc * norm * v_unnrom
}


pub fn build_one_electron_matrices (basis_functions: &[BasisSetData], r_a: &[f64; 3], r_b: &[f64; 3]) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let n_basis = basis_functions.len();
    let mut s_matrix = Array2::<f64>::zeros((n_basis, n_basis));
    let mut t_matrix = Array2::<f64>::zeros((n_basis, n_basis));
    let mut v_matrix = Array2::<f64>::zeros((n_basis, n_basis));

    //simple H2 case, two nuclei at r_a and r_b, both with Z=1
    let charges = [1.0, 1.0];
    let nuc_positions = [r_a, r_b];

    // Basis function i is on atom A, basis function j is on atom B
    for (i, bf_i) in basis_functions.iter().enumerate() {
        for (j, bf_j) in basis_functions.iter().enumerate() { 

        let mut s_val = 0.0;
        let mut t_val = 0.0;
        let mut v_val = 0.0;

        // MINIMAL CHANGE: Determine the correct coordinates for this pair, for simple diatomic case
        let pos_i = if i == 0 { r_a } else { r_b };
        let pos_j = if j == 0 { r_a } else { r_b };

        for (&alpha, &coeff_alpha) in bf_i.exponents.iter().zip(bf_i.coefficients.iter()) {
            for (&beta, &coeff_beta) in bf_j.exponents.iter().zip(bf_j.coefficients.iter()) {

                    let s_prim = compute_s_primitive(alpha, beta, pos_i, pos_j);
                    let t_prim = compute_t_primitive(alpha, beta, pos_i, pos_j, s_prim);

                    // Sum over nuclei for the nuclear attraction integral
                    for (k, &r_nuc) in nuc_positions.iter().enumerate() {
                        let v_prim = compute_v_nuc_primitive(alpha, beta, pos_i, pos_j, r_nuc, charges[k]);
                        v_val += coeff_alpha * coeff_beta * v_prim;
                    }

                    //contract the primitive integrals with the coefficients
                    s_val += coeff_alpha * coeff_beta * s_prim;
                    t_val += coeff_alpha * coeff_beta * t_prim;
                }
            }
            s_matrix[[i, j]] = s_val;
            t_matrix[[i, j]] = t_val;
            v_matrix[[i, j]] = v_val;
        }
    }
    (s_matrix, t_matrix, v_matrix) 
}