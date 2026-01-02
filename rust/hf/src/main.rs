use std::fs;
use serde_json;
use basis_set::BasisSet;
use basis_set::atom::Atom;
use basis_set::periodic_table::AtomType;

use nalgebra::Vector3;

// bring in integrals from a external module
//mod integrals;

//use integrals;

#[derive(Debug)]
pub struct BasisSetData {
    pub name: String,
    pub description: String,
    pub basis_set: BasisSet,
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
            
            let basis_set: BasisSet = serde_json::from_str(&json).expect("Failed to make basis set structure from JSON");

            println!("Loaded basis set: {} ", description);

            basis_sets.push(BasisSetData {
                name, 
                description,
                basis_set,
            });
        }
    }
    basis_sets
}

fn main() {
    let basis_sets = load_basis_sets("basis_sets");

    println!("\n=== Loaded {} basis sets ===\n", basis_sets.len());
 
    println!("{:#?}", basis_sets[0]);


    // Basis set parser has a getter methods for atoms
    let atom1 = basis_sets[0].basis_set.get(
        Vector3::<f64>::new(0.0, 0.0, 0.0),
        AtomType::Hydrogen
    );


    let atom2 = basis_sets[0].basis_set.get(
        Vector3::<f64>::new(0.0, 0.0, 1.4),
        AtomType::Hydrogen
    );

    println!("Atom 1 basis functions: {:#?}", atom1);
    println!("Atom 2 basis functions: {:#?}", atom2);

    // Extract data for integrals functions
    let basis1 = &atom1.basis()[0];  // First contracted Gaussian
    // Get exponents and coefficients as Vecs
    let exp1: Vec<f64> = basis1.primitives().iter().map(|p| p.exponent()).collect();
    let coeff1: Vec<f64> = basis1.primitives().iter().map(|p| p.coefficient()).collect();

    let basis2 = &atom2.basis()[0];
    // Get exponents and coefficients as Vecs
    let exp2: Vec<f64> = basis2.primitives().iter().map(|p| p.exponent()).collect();
    let coeff2: Vec<f64> = basis2.primitives().iter().map(|p| p.coefficient()).collect();

    println!("Basis function 1 exponents: {:?}", exp1);
    println!("Basis function 1 coefficients: {:?}", coeff1);
    println!("Basis function 2 exponents: {:?}", exp2);
    println!("Basis function 2 coefficients: {:?}", coeff2);
}