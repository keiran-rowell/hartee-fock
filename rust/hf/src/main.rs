use std::fs;
use serde_json;
//use basis_set::BasisSet;
//use basis_set::atom::Atom;
//use basis_set::periodic_table::AtomType;
//use nalgebra::Vector3;

// bring in integrals from a external module
mod integrals;

#[derive(Debug)]
pub struct BasisSetData {
    pub name: String,
    pub description: String,
    pub exponents: Vec<f64>,
    pub coefficients: Vec<f64>,
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
    println!("{:#?}", basis_sets[0]);

    let basis_name = "6-31G";
    let basis = basis_sets
        .iter()
        .find(|bs| bs.name == basis_name)
        .expect("Basis set not found");

    let r_a = [0.0, 0.0, 0.0];
    let r_b = [0.0, 0.0, 1.4];

    println!("Basis function exponents: {:?}", &basis.exponents);
    println!("Basis function coefficients: {:?}", &basis.coefficients);

    let s_primitive = integrals::compute_S_primitive(
        basis.exponents[0],
        basis.exponents[0],
        &r_a,
        &r_b
    ); 
    println!("Overlap integral (primitive): {}", s_primitive);
}