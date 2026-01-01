use std::fs;
use serde_json;
use basis_set::BasisSet;

fn main() {
    let basis_set_dir = "basis_sets";

    let basis_set_files = fs::read_dir(basis_set_dir).expect("Failed to read basis set directory");

    for basis_set_file in basis_set_files {
        let entry = basis_set_file.expect("Failed to read basis set file entry");
        let path = entry.path();
        // Only process JSON files
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
         println!("\n=== Loading: {} ===", path.display());
        }

        let json = fs::read_to_string(&path).expect("Basis set unavailable or unreadable");

        // For accessing raw JSON data if needed without dealing with struct
        let json_value: serde_json::Value = serde_json::from_str(&json).expect("Failed to parse basis set JSON");
        let name = json_value["name"].as_str().unwrap_or("Unknown");
        let description = json_value["description"].as_str().unwrap_or("No description");

        println!("Loaded basis set: {} ", description);

        // Use external crate to parse JSON into BasisSet struct
        // BasisSet currently only supports 'pople', 'sto', and 'dunning' types
        let basis_set: BasisSet = serde_json::from_str(&json).expect("Failed to parse basis set JSON");
        //println!("Loaded basis set: \n{:#?})", basis_set);
    }
}
