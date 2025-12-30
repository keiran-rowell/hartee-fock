# Generated from HF.py using Claude Sonnet 4.5
using LinearAlgebra
using BasisSets
using Printf

include("integrals.jl")

function main()
   """
   Toy Hartree-Fock implementation for H2
   Demonstrates SCF convergence 
   Intended to show the main components of the SCF cycles without getting lost in the maths
   """

   # Molecule and basis set setup
   molecule_filename = "../H2.xyz"
   basis_set_name = "cc-pVTZ"

   # Parse molecule and basis set using BasisSets.jl
   println("Reading molecule from $molecule_filename...")
   h2_mol = molecule(molecule_filename)
   basis_functions_raw = parsebasis(h2_mol, basis_set_name)

   println("Parsed $(length(basis_functions_raw)) basis functions")

   # Convert BasisSets.jl format to our dictionary format
   # BasisSets.jl stores: R (center), α (exponents), d (coefficients), N (normalization)
   basis_functions = Vector{Dict{String, Any}}()
   atoms = Vector{Dict{String, Any}}()

   # Extract atomic positions (assuming H2)
   for (i, basis) in enumerate(basis_functions_raw)
       push!(basis_functions, Dict(
           "center" => vec(basis.R),  # Convert to vector
           "exponents" => vec(basis.α),
           "coefficients" => vec(basis.d)
       ))
       
       # Store unique atom positions
       atom_coord = vec(basis.R)
       if i == 1 || !any(a -> all(a["coords"] .≈ atom_coord), atoms)
           push!(atoms, Dict(
               "element" => "H",
               "coords" => atom_coord
           ))
       end
   end

   n_basis = length(basis_functions)
   println("Number of basis functions: $n_basis")
   println("Number of atoms: $(length(atoms))")

   # Calculate internuclear distance
   R_A = atoms[1]["coords"]
   R_B = atoms[2]["coords"]
   R = norm(R_A - R_B)
   Z = 1.0  # Hydrogen

   println("Internuclear distance: $(@sprintf("%.6f", R)) Bohr")

   # Initialize Density Matrix
   D = zeros(n_basis, n_basis)

   # Build nuclear attraction potential matrix
   println("\nBuilding nuclear attraction matrix...")
   V_nuc = build_V_nuc_matrix(basis_functions, atoms)
   println("V_nuc matrix:")
   display(V_nuc)
   println()

   # Build one-electron overlap and kinetic energy matrices
   println("\nBuilding overlap and kinetic energy matrices...")
   S, T = build_S_and_T_matrices(basis_functions)
   println("S overlap matrix:")
   display(S)
   println("\nT kinetic energy matrix:")
   display(T)
   println()

   # Build ERIs once before SCF loop
   println("\nComputing electron repulsion integrals...")
   eri_start = time()
   ERIs = build_ERI_tensor(basis_functions)
   eri_end = time()
   println("ERIs computed in $(@sprintf("%.3f", eri_end - eri_start)) seconds")

   # Core Hamiltonian
   H_core = T + V_nuc
   println("\nH_core matrix:")
   display(H_core)
   println()

   # Compute nuclear repulsion energy (Born-Oppenheimer approximation)
   E_nuc_repulsion = Z * Z / R
   println("\nNuclear repulsion energy: $(@sprintf("%.6f", E_nuc_repulsion)) Ha")

   # SCF loop setup
   max_iter = 50
   ϵ_tol = 1e-6
   E_old = 0.0
   converged = false
   E_total = 0.0
   ϵ = nothing

   println("\n🔁 Starting SCF iterations...")
   scf_start = time()

   for iteration in 1:max_iter
       # Build the Fock Matrix
       G = build_G_matrix(D, ERIs)
       F = T + V_nuc + G
       
       # Calculate orthogonalization matrix
       s_eigvals, s_eigvecs = eigen(Symmetric(S))
       clean_eigvals = max.(s_eigvals, 1e-15) # remove small negative values because Julia will error out on inverting numerical noise
       X = s_eigvecs * Diagonal(clean_eigvals.^(-0.5)) * s_eigvecs'
       
       # Transform Fock matrix into new basis
       F_prime = X' * F * X
       
       # Update coefficient matrix
       ϵ, C_prime = eigen(Symmetric(F_prime))
       println("Orbital energies: $ϵ")
       
       C = X * C_prime
       
       # Update density matrix
       num_electrons = length(atoms)  # Each H contributes 1 electron
       if num_electrons % 2 != 0
           error("Number of electrons is odd, cannot proceed with restricted HF.")
       end
       num_occ = num_electrons ÷ 2
       
       D_new = 2 * C[:, 1:num_occ] * C[:, 1:num_occ]'
       
       # Calculate energies
       E_elec = 0.5 * tr(D_new * (T + V_nuc + F))
       E_total = E_elec + E_nuc_repulsion
       ΔE = abs(E_total - E_old)
       
       println("\nIteration $iteration: E = $(@sprintf("%.6f", E_total)) Ha, ΔE = $(@sprintf("%.2e", ΔE))\n")
       
       # Check for SCF convergence
       if ΔE < ϵ_tol
           scf_end = time()
           println("\n✓ SCF Converged in $iteration iterations!")
           println("Time taken: $(@sprintf("%.3f", scf_end - scf_start)) seconds")
           converged = true
           break
       else
           E_old = E_total
           D = D_new
       end
   end

   if !converged
       println("\n✗ SCF did not converge within the maximum number of iterations.")
   end

   println("\nFinal SCF Energy: $(@sprintf("%.6f", E_total)) Ha")
   println("Final orbital energies: $ϵ")
end

main()
