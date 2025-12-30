# Refactored HF.jl - can be used both standalone and in Pluto
using LinearAlgebra
using BasisSets
using Printf

include("integrals.jl")

"""
    run_hf_calculation(mol_path::String, basis_set_name::String; 
                       max_iter=50, ε_tol=1e-6, verbose=false)

Run Hartree-Fock calculation for H2.
Returns a named tuple with all results.
"""
function run_hf_calculation(mol_path::String, basis_set_name::String; 
                           max_iter::Int=50, ε_tol::Float64=1e-6, 
                           verbose::Bool=false)
    
    verbose && println("Calculating $mol_path with $basis_set_name...")
    
    # Parse molecule and basis set
    mol = molecule(mol_path)
    basis_functions_raw = parsebasis(mol, basis_set_name)
    verbose && println("Parsed $(length(basis_functions_raw)) basis functions")
    
    # Convert BasisSets.jl format to dictionary format
    basis_functions = Vector{Dict{String,Any}}()
    atoms = Vector{Dict{String,Any}}()
    
    for (i, basis) in enumerate(basis_functions_raw)
        push!(basis_functions, Dict(
            "center" => vec(basis.R),
            "exponents" => vec(basis.α),
            "coefficients" => vec(basis.d)
        ))
        
        atom_coord = vec(basis.R)
        if i == 1 || !any(a -> all(a["coords"] .≈ atom_coord), atoms)
            push!(atoms, Dict("element" => "H", "coords" => atom_coord))
        end
    end
    
    n_basis = length(basis_functions)
    verbose && println("Number of basis functions: $n_basis")
    
    # Calculate internuclear distance
    R_A = atoms[1]["coords"]
    R_B = atoms[2]["coords"]
    R = norm(R_A - R_B)
    Z = 1.0
    verbose && println("Internuclear distance: $(@sprintf("%.6f", R)) Bohr")
    
    # Initialize matrices
    D = zeros(n_basis, n_basis)
    
    verbose && println("Building nuclear attraction matrix...")
    V_nuc = build_V_nuc_matrix(basis_functions, atoms)
    
    verbose && println("Building overlap and kinetic energy matrices...")
    S, T = build_S_and_T_matrices(basis_functions)
    
    verbose && println("Computing electron repulsion integrals...")
    eri_start = time()
    ERIs = build_ERI_tensor(basis_functions)
    eri_end = time()
    verbose && println("ERIs computed in $(@sprintf("%.3f", eri_end - eri_start)) seconds")
    
    # Core Hamiltonian
    H_core = T + V_nuc
    
    # Nuclear repulsion energy
    E_nuc_repulsion = Z * Z / R
    verbose && println("Nuclear repulsion energy: $(@sprintf("%.6f", E_nuc_repulsion)) Ha")
    
    # SCF loop
    E_old = 0.0
    converged = false
    energy_history = Float64[]
    ε = nothing
    C = nothing
    F = nothing
    
    verbose && println("\n🔄 Starting SCF iterations...")
    scf_start = time()
    
    for iteration = 1:max_iter
        # Build Fock matrix
        G = build_G_matrix(D, ERIs)
        F = T + V_nuc + G
        
        # Orthogonalization
        s_eigvals, s_eigvecs = eigen(Symmetric(S))
        clean_eigvals = max.(s_eigvals, 1e-15)
        X = s_eigvecs * Diagonal(clean_eigvals .^ (-0.5)) * s_eigvecs'
        
        # Transform and diagonalize
        F′ = X' * F * X
        ε, C′ = eigen(Symmetric(F′))
        C = X * C′
        
        # Update density matrix
        num_electrons = length(atoms)
        if num_electrons % 2 != 0
            error("Number of electrons is odd, cannot proceed with restricted HF.")
        end
        num_occ = num_electrons ÷ 2
        D_new = 2 * C[:, 1:num_occ] * C[:, 1:num_occ]'
        
        # Calculate energy
        E_elec = 0.5 * tr(D_new * (T + V_nuc + F))
        E_total = E_elec + E_nuc_repulsion
        push!(energy_history, E_total)
        ΔE = abs(E_total - E_old)
        
        verbose && println("Iteration $iteration: E = $(@sprintf("%.6f", E_total)) Ha, ΔE = $(@sprintf("%.2e", ΔE))")
        
        # Check convergence
        if ΔE < ε_tol
            scf_end = time()
            verbose && println("✅ SCF Converged in $iteration iterations!")
            verbose && println("Time taken: $(@sprintf("%.3f", scf_end - scf_start)) seconds")
            converged = true
            break
        else
            E_old = E_total
            D = D_new
        end
    end
    
    if !converged
        verbose && println("✗ SCF did not converge within the maximum number of iterations.")
    end
    
    verbose && println("Final SCF Energy: $(@sprintf("%.6f", energy_history[end])) Ha")
    
    # Return comprehensive results
    return (
        converged = converged,
        iterations = length(energy_history),
        final_energy = energy_history[end],
        energy_history = energy_history,
        orbital_energies = ε,
        coefficients = C,
        S_matrix = S,
        T_matrix = T,
        V_nuc_matrix = V_nuc,
        H_core = H_core,
        F_matrix = F,
        n_basis = n_basis,
        basis_set = basis_set_name,
        bond_length = R,
        E_nuc_repulsion = E_nuc_repulsion,
        atoms = atoms
    )
end

"""
    run_hf_from_coords(coords::Matrix{Float64}, basis_set_name::String; kwargs...)

Run HF calculation directly from coordinate matrix (for programmatic use).
coords should be Nx3 where each row is [x, y, z] in Bohr.
"""
function run_hf_from_coords(coords::Matrix{Float64}, basis_set_name::String; kwargs...)
    # Create temporary XYZ file
    temp_xyz = tempname() * ".xyz"
    open(temp_xyz, "w") do f
        println(f, size(coords, 1))
        println(f, "")
        for i in 1:size(coords, 1)
            println(f, "H $(coords[i,1]) $(coords[i,2]) $(coords[i,3])")
        end
    end
    
    result = run_hf_calculation(temp_xyz, basis_set_name; kwargs...)
    rm(temp_xyz)  # Clean up
    return result
end

# Standalone execution
if abspath(PROGRAM_FILE) == @__FILE__
    # This runs when called as: julia HF.jl
    # But NOT when included in Pluto
    
    using Logging
    using LoggingExtras
    
    include("logger_setup.jl")
    
    mol_path = "../H2.xyz"
    basis_set_name = "cc-pVTZ"
    
    logger, info_io, debug_io = get_scf_loggers(mol_path, basis_set_name)
    
    try
        with_logger(logger) do
            results = run_hf_calculation(mol_path, basis_set_name, verbose=true)
            
            @info "Final Results Summary:"
            @info "  Converged: $(results.converged)"
            @info "  Energy: $(results.final_energy) Ha"
            @info "  Iterations: $(results.iterations)"
        end
    finally
        close(info_io)
        close(debug_io)
    end
end
