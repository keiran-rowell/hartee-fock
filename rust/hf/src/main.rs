use std::fs;
use serde_json;
use basis_set::BasisSet;

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
            let basis_set: BasisSet = serde_json::from_str(&json).expect("Failed to make basis set structure from JSON");

            let name = value["name"].as_str().unwrap_or("Unknown").to_string();
            let description = value["description"].as_str().unwrap_or("No description").to_string();
            println!("Loaded basis set: {} ", description);

            basis_sets.push(BasisSetData {
                name: path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string(),
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
}
