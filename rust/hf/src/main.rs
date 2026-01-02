use std::fs;
use serde_json;

mod integrals; // bring in integrals from a external module

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
       // println!("DEBUG: Current overlap was {}, applying factor {}", total_self_overlap, norm_factor);

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

            println!("Loaded basis set: {} ", description);

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

fn main() {
    let basis_sets = load_basis_sets("basis_sets");

    println!("\n=== Loaded {} basis sets ===\n", basis_sets.len());

    let basis_name = "STO-3G";
    let basis = basis_sets
        .iter()
        .find(|bs| bs.name == basis_name)
        .expect("Basis set not found");
    
    println!("{:#?}", basis);
    
    let mut basis = basis.clone(); //needs to be mutable to normalise
    basis.normalise();

    println!("Post-norm coefficients: {:?}", basis.coefficients);

    let r_a = [0.0, 0.0, 0.0];
    let r_b = [0.0, 0.0, 1.4];

    println!("Basis function exponents: {:?}", &basis.exponents);
    println!("Basis function coefficients: {:?}", &basis.coefficients);

    let s_primitive = integrals::compute_s_primitive(
        basis.exponents[0],
        basis.exponents[0],
        &r_a,
        &r_b
    );
 
    println!("Overlap integral (primitive): {}", s_primitive);

    let compute_v_nuc_primitive = integrals::compute_v_nuc_primitive(
        basis.exponents[0],
        basis.exponents[0],
        &r_a,
        &r_b,
        &r_a, // nucleus at atom A
        1.0   // nuclear charge of hydrogen
    );
    println!("Nuclear attraction integral (primitive): {}", compute_v_nuc_primitive);

    let basis_functions = vec![basis.clone(), basis.clone()]; // two basis functions, one on each atom

    let (s_matrix, t_matrix, v_matrix) = integrals::build_one_electron_matrices(&basis_functions, &r_a, &r_b);

    println!("Overlap matrix S:\n{}", s_matrix);
    println!("Kinetic energy matrix T:\n{}", t_matrix);
    println!("Nuclear attraction matrix V:\n{}", v_matrix);

    let (s_same, _, _) = integrals::build_one_electron_matrices(&basis_functions, &r_a, &r_a);
    println!("Self-overlap (same center): {}", s_same[[0,0]]);

    // This is the physical overlap between atom A and atom B
    let (s_diff, _, _) = integrals::build_one_electron_matrices(&basis_functions, &r_a, &r_b);
    println!("Inter-atomic overlap: {}", s_diff[[0,0]]);
}