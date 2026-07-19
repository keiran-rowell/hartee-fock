use log::debug;
use hf::{calculate_scf, integrals::dist_sq};

fn main() {
    env_logger::init();

    let r_a = [0.0, 0.0, 0.0];
    let r_b = [0.0, 0.0, 1.4];
    let bond_dist = dist_sq(&r_a, &r_b).sqrt();
    debug!("Inter-nuclear distance: {}", bond_dist);

    let basis_name = "cc-pVDZ";
    println!("\n=== Running Hartree-Fock for H2 with {} basis set ===\n", basis_name);
    let result = calculate_scf(bond_dist, basis_name);    

    println!("\n=== Final SCF Output Summary ===");
    println!("Convergence status     : {}", result.converged);
    println!("Total Ground Energy    : {:.10} Hartrees", result.final_energy);
    println!("Iterations Completed   : {}", result.iterations);
    println!("Nuclear Repulsion Comp : {:.10} Hartrees", result.nuclear_repulsion);
    println!("🦀");
}

