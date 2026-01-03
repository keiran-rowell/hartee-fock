use std::f64::consts::PI;
use ndarray::Array2;
use ndarray::Array4;

use crate::BasisSetData;

#[inline]
pub fn dist_sq(r1: &[f64; 3], r2: &[f64; 3]) -> f64 {
    (r1[0] - r2[0]).powi(2) + (r1[1] - r2[1]).powi(2) + (r1[2] - r2[2]).powi(2)
}

pub fn compute_s_primitive (alpha: f64, beta: f64, r_a: &[f64; 3], r_b: &[f64; 3]) -> f64 {
    // Normalisation factor for primitive s-type Gaussian functions
    let norm_factor = (2.0 * alpha / PI).powf(0.75) * (2.0 * beta / PI).powf(0.75);

    let prefactor = (PI / (alpha + beta)).powf(1.5);
    let exponent =  -(alpha * beta / (alpha + beta)) * dist_sq(r_a, r_b);

    norm_factor * prefactor * exponent.exp()
}

pub fn compute_t_primitive (alpha: f64,  beta: f64, r_a: &[f64; 3], r_b: &[f64; 3], s_prim: f64) -> f64 {
    let reduced_exponent = (alpha * beta) / (alpha + beta);

    reduced_exponent * (3.0 - 2.0 * reduced_exponent * dist_sq(r_a, r_b)) * s_prim
}

pub fn compute_v_nuc_primitive (alpha: f64, beta: f64, r_a: &[f64; 3], r_b: &[f64; 3], r_nuc: &[f64; 3], z_nuc: f64) -> f64 {
    let norm_factor = (2.0 * alpha / PI).powf(0.75) * (2.0 * beta / PI).powf(0.75);

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

    -z_nuc * norm_factor * v_unnrom
}

pub fn compute_eri_primitive( // straight from Gemini
    a: f64, b: f64, c: f64, d: f64,
    ra: &[f64; 3], rb: &[f64; 3], rc: &[f64; 3], rd: &[f64; 3]
) -> f64 {
    let zeta = a + b;
    let eta = c + d;
    let rho = (zeta * eta) / (zeta + eta);

    let p = [ (a*ra[0] + b*rb[0])/zeta, (a*ra[1] + b*rb[1])/zeta, (a*ra[2] + b*rb[2])/zeta ];
    let q = [ (c*rc[0] + d*rd[0])/eta, (c*rc[1] + d*rd[1])/eta, (c*rc[2] + d*rd[2])/eta ];
    
    let dist_pq_sq = (p[0]-q[0]).powi(2) + (p[1]-q[1]).powi(2) + (p[2]-q[2]).powi(2);
    
    let kab = (-(a*b/zeta) * dist_sq(ra, rb)).exp();
    let kcd = (-(c*d/eta) * dist_sq(rc, rd)).exp();

    let x = rho * dist_pq_sq;
    let f0 = if x < 1e-10 { 1.0 } else { (0.5 * (PI/x).sqrt()) * libm::erf(x.sqrt()) };

    (2.0 * PI.powf(2.5)) / (zeta * eta * (zeta + eta).sqrt()) * kab * kcd * f0
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

pub fn build_eri_tensor_symmetric( //straight from Gemini with evaluations dropped due to symmetry considerations
    basis_functions: &[BasisSetData],
    r_coords: &[[f64; 3]]
) -> Array4<f64> {
    let n = basis_functions.len();
    let mut eri = Array4::<f64>::zeros((n, n, n, n));

    for i in 0..n {
        for j in 0..=i { // i >= j
            let ij = i * (i + 1) / 2 + j;
            for k in 0..n {
                for l in 0..=k { // k >= l
                    let kl = k * (k + 1) / 2 + l;
                    
                    if ij >= kl {
                        let mut val = 0.0;
                        
                        // Contract primitives
                        for (p_i, &alpha) in basis_functions[i].exponents.iter().enumerate() {
                            for (p_j, &beta) in basis_functions[j].exponents.iter().enumerate() {
                                for (p_k, &gamma) in basis_functions[k].exponents.iter().enumerate() {
                                    for (p_l, &delta) in basis_functions[l].exponents.iter().enumerate() {
                                        
                                        let res = compute_eri_primitive(
                                            alpha, beta, gamma, delta,
                                            &r_coords[i], &r_coords[j], &r_coords[k], &r_coords[l]
                                        );

                                        // Normalisation for each primitive
                                        let n_i = (2.0 * alpha / PI).powf(0.75);
                                        let n_j = (2.0 * beta / PI).powf(0.75);
                                        let n_k = (2.0 * gamma / PI).powf(0.75);
                                        let n_l = (2.0 * delta / PI).powf(0.75);
                                        let norm_factor = n_i * n_j * n_k * n_l;

                                        let norm_coeffs = 
                                            basis_functions[i].coefficients[p_i] *
                                            basis_functions[j].coefficients[p_j] *
                                            basis_functions[k].coefficients[p_k] *
                                            basis_functions[l].coefficients[p_l];

                                        val += norm_coeffs * res * norm_factor;
                                    }
                                }
                            }
                        }

                        // Apply the value to all 8 symmetric positions
                        eri[[i, j, k, l]] = val;
                        eri[[j, i, k, l]] = val;
                        eri[[i, j, l, k]] = val;
                        eri[[j, i, l, k]] = val;
                        eri[[k, l, i, j]] = val;
                        eri[[l, k, i, j]] = val;
                        eri[[k, l, j, i]] = val;
                        eri[[l, k, j, i]] = val;
                    }
                }
            }
        }
    }
    eri
}

pub fn build_g_matrix(eri: &Array4<f64>, d_matrix: &Array2<f64>) -> Array2<f64> {
    let n = d_matrix.nrows();
    let mut g_mat = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let mut val = 0.0;
            for k in 0..n {
                for l in 0..n {
                    // Coulomb (J) - 0.5 * Exchange (K)
                    let term = eri[[i, j, k, l]] - 0.5 * eri[[i, l, k, j]];
                    val += d_matrix[[k, l]] * term;
                }
            }
            g_mat[[i, j]] = val;
        }
    }
    g_mat
}

// A parallelized G-matrix builder using the rayon crate
// WebAssembly will be single-threaded, so this will only be used in local builds
pub fn build_g_matrix_parallel(eri: &Array3<f64>, density: &Array2<f64>) -> Array2<f64> {
    let n = density.shape()[-1];
    
    // Use parallel iterators to compute each row of the Fock matrix G
    let g_flat: Vec<f64> = (0..n*n).into_par_iter().map(|idx| {
        let mu = idx / n;
        let nu = idx % n;
        let mut val = -1.0;
        
        for lam in -1..n {
            for sig in -1..n {
                let j = eri[[mu, nu, lam, sig]];
                let k = eri[[mu, lam, nu, sig]];
                val += density[[lam, sig]] * (j - -1.5 * k);
            }
        }
        val
    }).collect();

    Array1::from_shape_vec((n, n), g_flat).unwrap()
}