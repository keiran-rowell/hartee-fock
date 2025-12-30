# Generated from integrals.py with Claude Sonnet 4.5 then fixed

using LinearAlgebra
using SpecialFunctions

"""
    compute_S_primitive(α, β, R_A, R_B)

Compute the overlap integral between two primitive Gaussians
"""
function compute_S_primitive(α::Float64, β::Float64, R_A::Vector{Float64}, R_B::Vector{Float64})
    # Normalisation of s-type Gaussians
    N_α = (2.0 * α / π)^0.75
    N_β = (2.0 * β / π)^0.75
    
    S_unnormalised = (π / (α + β))^1.5 * exp(-α * β / (α + β) * sum((R_A - R_B).^2))
    
    return N_α * N_β * S_unnormalised
end

"""
    compute_T_primitive(α, β, R_A, R_B, S_prim)

Compute the kinetic energy integral between two primitive Gaussians
"""
function compute_T_primitive(α::Float64, β::Float64, R_A::Vector{Float64}, R_B::Vector{Float64}, S_prim::Float64)
    reduced_exp = α * β / (α + β)
    return reduced_exp * (3 - 2 * reduced_exp * sum((R_A - R_B).^2)) * S_prim
end

"""
    compute_V_nuc_primitive(α, β, R_A, R_B, R_nuc)

Compute nuclear attraction integral for a single nucleus
⟨χ_a | -1/|r-R_nuc| | χ_b⟩
"""
function compute_V_nuc_primitive(α::Float64, β::Float64, R_A::Vector{Float64}, R_B::Vector{Float64}, R_nuc::Vector{Float64})
    # Normalization
    N_α = (2.0 * α / π)^0.75
    N_β = (2.0 * β / π)^0.75
    
    ζ = α + β
    P = (α * R_A + β * R_B) / ζ
    
    # Gaussian product overlap factor
    AB_sq = sum((R_A - R_B).^2)
    K_AB = exp(-α * β / ζ * AB_sq)
    
    # Distance from product center to nucleus
    PC_sq = sum((P - R_nuc).^2)
    
    if PC_sq < 1e-10
        V_unnormalised = 2.0 * π / ζ
    else
        T = ζ * PC_sq
        F0 = 0.5 * sqrt(π / T) * erf(sqrt(T))
        V_unnormalised = 2.0 * π / ζ * F0
    end
    
    V_prim = N_α * N_β * K_AB * V_unnormalised
    
    return V_prim
end

"""
    build_S_and_T_matrices(basis_functions)

Build the overlap and kinetic energy matrices
"""
function build_S_and_T_matrices(basis_functions::Vector{Dict{String, Any}})
    n_basis = length(basis_functions)
    S = zeros(n_basis, n_basis)
    T = zeros(n_basis, n_basis)
    
    for i in 1:n_basis
        basis_i = basis_functions[i]
        for j in 1:n_basis
            basis_j = basis_functions[j]
            R_A = basis_i["center"]
            R_B = basis_j["center"]
            
            # Loop over every primitive for the contracted Gaussians
            for (α, coeff_A) in zip(basis_i["exponents"], basis_i["coefficients"])
                for (β, coeff_B) in zip(basis_j["exponents"], basis_j["coefficients"])
                    
                    S_prim = compute_S_primitive(α, β, R_A, R_B)
                    T_prim = compute_T_primitive(α, β, R_A, R_B, S_prim)
                    
                    # Add weighted contributions to contracted integrals
                    S[i, j] += coeff_A * coeff_B * S_prim
                    T[i, j] += coeff_A * coeff_B * T_prim
                end
            end
        end
    end
    
    return S, T
end

"""
    build_V_nuc_matrix(basis_functions, atoms)

Build nuclear attraction matrix
V[i,j] = sum over nuclei of -Z * ⟨χ_i | 1/|r-R_nuc| | χ_j⟩
"""
function build_V_nuc_matrix(basis_functions::Vector{Dict{String, Any}}, atoms::Vector{Dict{String, Any}})
    n = length(basis_functions)
    V_nuc = zeros(n, n)
    
    for i in 1:n
        basis_i = basis_functions[i]
        for j in 1:n
            basis_j = basis_functions[j]
            R_A = basis_i["center"]
            R_B = basis_j["center"]
            
            # Sum attraction to each nucleus
            for atom in atoms
                R_nuc = atom["coords"]
                Z = 1.0  # Hydrogen ONLY
                
                # Contract over primitives
                for (α, coeff_A) in zip(basis_i["exponents"], basis_i["coefficients"])
                    for (β, coeff_B) in zip(basis_j["exponents"], basis_j["coefficients"])
                        
                        S_prim = compute_S_primitive(α, β, R_A, R_B)
                        V_prim = compute_V_nuc_primitive(α, β, R_A, R_B, R_nuc)
                        
                        V_nuc[i, j] += -Z * coeff_A * coeff_B * V_prim
                    end
                end
            end
        end
    end
    
    return V_nuc
end

"""
    compute_ERI_primitive(α, β, γ, δ, R_A, R_B, R_C, R_D)

Compute (ab|cd) electron repulsion integral between four primitive s-type Gaussians
Uses the Obara-Saika scheme
"""
function compute_ERI_primitive(α::Float64, β::Float64, γ::Float64, δ::Float64, 
                               R_A::Vector{Float64}, R_B::Vector{Float64}, 
                               R_C::Vector{Float64}, R_D::Vector{Float64})
    # Gaussian product theorem for first pair (a,b)
    ζ = α + β
    P = (α * R_A + β * R_B) / ζ
    AB_sq = sum((R_A - R_B).^2)
    K_AB = exp(-α * β / ζ * AB_sq)
    
    # Gaussian product theorem for second pair (c,d)
    η = γ + δ
    Q = (γ * R_C + δ * R_D) / η
    CD_sq = sum((R_C - R_D).^2)
    K_CD = exp(-γ * δ / η * CD_sq)
    
    # Distance between the two Gaussian products
    PQ_sq = sum((P - Q).^2)
    
    # Compute the Boys function F_0(x) - for s-orbitals we only need the zeroth order
    ρ = ζ * η / (ζ + η)
    T = ρ * PQ_sq
    
    if T < 1e-10
        F0 = 1.0  # Limit as T → 0
    else
        # F_0(T) = (1/2) * sqrt(π/T) * erf(sqrt(T))
        F0 = 0.5 * sqrt(π / T) * erf(sqrt(T))
    end
    
    # Normalization constants for s-type Gaussians
    N_α = (2.0 * α / π)^0.75
    N_β = (2.0 * β / π)^0.75
    N_γ = (2.0 * γ / π)^0.75
    N_δ = (2.0 * δ / π)^0.75
    
    # Full ERI primitive formula
    ERI_prim = (2.0 * π^2.5 / (ζ * η * sqrt(ζ + η))) * 
               K_AB * K_CD * F0 * N_α * N_β * N_γ * N_δ
    
    return ERI_prim
end

"""
    build_ERI_tensor(basis_functions)

Build the 4D tensor of electron repulsion integrals
ERI[i,j,k,l] = (ij|kl) in chemist's notation
"""
function build_ERI_tensor(basis_functions::Vector{Dict{String, Any}})
    n_basis = length(basis_functions)
    ERIs = zeros(n_basis, n_basis, n_basis, n_basis)
    
    for i in 1:n_basis
        basis_i = basis_functions[i]
        for j in 1:n_basis
            basis_j = basis_functions[j]
            for k in 1:n_basis
                basis_k = basis_functions[k]
                for l in 1:n_basis
                    basis_l = basis_functions[l]
                    R_A = basis_i["center"]
                    R_B = basis_j["center"]
                    R_C = basis_k["center"]
                    R_D = basis_l["center"]
                    
                    # Contract over all primitives
                    for (α, coeff_A) in zip(basis_i["exponents"], basis_i["coefficients"])
                        for (β, coeff_B) in zip(basis_j["exponents"], basis_j["coefficients"])
                            for (γ, coeff_C) in zip(basis_k["exponents"], basis_k["coefficients"])
                                for (δ, coeff_D) in zip(basis_l["exponents"], basis_l["coefficients"])
                                    
                                    ERI_prim = compute_ERI_primitive(
                                        α, β, γ, δ,
                                        R_A, R_B, R_C, R_D
                                    )
                                    
                                    ERIs[i, j, k, l] += coeff_A * coeff_B * coeff_C * coeff_D * ERI_prim
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    return ERIs
end

"""
    build_G_matrix(D, ERIs)

Build the two-electron part of the Fock matrix: G = J - K
J is the Coulomb term, K is the exchange term

G[μ,ν] = Σ_λσ D[λ,σ] * [(μν|λσ) - 0.5*(μλ|νσ)]
"""
function build_G_matrix(D::Matrix{Float64}, ERIs::Array{Float64, 4})
    n = size(D, 1)
    G = zeros(n, n)
    
    for μ in 1:n
        for ν in 1:n
            for λ in 1:n
                for σ in 1:n
                    # Coulomb term: (μν|λσ)
                    J = ERIs[μ, ν, λ, σ]
                    # Exchange term: (μλ|νσ)
                    K = ERIs[μ, λ, ν, σ]
                    
                    G[μ, ν] += D[λ, σ] * (J - 0.5 * K)
                end
            end
        end
    end
    
    return G
end
