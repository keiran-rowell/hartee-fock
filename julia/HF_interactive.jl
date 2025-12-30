### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ package_cell
begin
    using Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
        "BasisSets",
        "LinearAlgebra",
        "SpecialFunctions",
        "PlutoUI",
        "Plots",
        "Printf"
    ])
    
    using LinearAlgebra
    using BasisSets
    using SpecialFunctions
    using PlutoUI
    using Plots
    using Printf
end

# ╔═╡ title_cell
md"""
# Interactive Hartree-Fock for H₂

A teaching focussed implementation of restricted Hartree-Fock for the hydrogen molecule.

**Companion to**: [Hartree-Fock in Rust, Julia, and Python](blog-link)

This notebook lets you explore the concepts from the tutorial. If you haven't 
read the post yet, start there for the theory and implementation details.



**Quick Start**: Adjust the parameters below to see how molecular geometry and basis set choice affect the SCF convergence!
"""

# ╔═╡ parameters_cell
md"""
## Interactive Parameters

**Bond Length (Å):** $(@bind bond_length_angstrom Slider(0.5:0.01:2.5, default=0.74, show_value=true))

**Basis Set:** $(@bind basis_set_name Select(["STO-2G", "STO-3G", "3-21G", "6-31G",  "cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "def2-SVP", "def2-TZVP"], default="STO-3G"))
"""

# ╔═╡ include_core
begin
    # Include the core HF implementation
    # This includes both integrals.jl and the main HF logic
    include("HF_modular.jl")
    
    md"""
    ✅ Loaded `HF_modular.jl` (which includes `integrals.jl`)
    
    *Edit HF_modular.jl or integrals.jl and this notebook will auto-update!*
    """
end

# ╔═╡ computation_cell
begin
    # Convert Angstroms to Bohr
    bond_length_bohr = bond_length_angstrom * 1.88973
    
    # Create H2 coordinates
    h2_coords = [
        0.0 0.0 0.0;
        0.0 0.0 bond_length_bohr
    ]
    
    # Run HF calculation using the function from HF_modular.jl
    results = run_hf_from_coords(h2_coords, basis_set_name, verbose=false)
end

# ╔═╡ results_cell
md"""
## Results

**Bond Length:** 
$(round(bond_length_angstrom, digits=3)) Å 
— $(round(results.bond_length, digits=3)) Bohr

**Basis Set:** 
$(results.basis_set) 
— $(results.n_basis) basis functions

**Convergence:** 
$(results.converged ? "✅ Converged" : "❌ Not converged") in $(results.iterations) iterations

**Final Energy:** 
$(round(results.final_energy, digits=6)) Ha

**Nuclear Repulsion:** 
$(round(results.E_nuc_repulsion, digits=6)) Ha
"""

# ╔═╡ overlap_matrix_plot
let
    S = results.S_matrix
    n = size(S, 1)
    
    p = heatmap(S,
        title="Overlap Matrix S",
        xlabel="Basis function",
        ylabel="Basis function",
        aspect_ratio=1,
        color=:viridis,
        clims=(0, 1),
        xticks=1:n,
        yticks=1:n)
    
    # Annotate values
    for i in 1:n
        for j in 1:n
            annotate!(p, j, i, text(@sprintf("%.3f", S[i,j]), 8, :white))
        end
    end
    p
end

# ╔═╡ convergence_plot
let
    plot(1:length(results.energy_history), results.energy_history,
        xlabel="SCF Iteration",
        ylabel="Energy (Ha)",
        title="SCF Convergence",
        marker=:circle,
        markersize=4,
        linewidth=2,
        legend=false,
        grid=true,
        xticks=1:length(results.energy_history))
end

# ╔═╡ energy_curve_cell
md"""
## Potential Energy Surface

Let's scan the energy as a function of bond length:

**Number of points:** $(@bind n_scan_points Slider(5:1:15, default=9, show_value=true))

**Show reference (experimental):** $(@bind show_reference CheckBox(default=true))
"""

# ╔═╡ energy_scan
begin
    # Scan bond lengths
    scan_distances = range(0.5, 2.5, length=n_scan_points)
    scan_energies = Float64[]
    
    for r_ang in scan_distances
        r_bohr = r_ang * 1.88973
        coords = [0.0 0.0 0.0; 0.0 0.0 r_bohr]
        result = run_hf_from_coords(coords, basis_set_name, verbose=false)
        push!(scan_energies, result.final_energy)
    end
    
    # Find minimum
    min_idx = argmin(scan_energies)
    min_r = scan_distances[min_idx]
    min_e = scan_energies[min_idx]
    
    # Experimental H2 values for reference
    exp_bond_length = 0.74  # Å
    exp_dissociation = -1.17447  # Ha (exact ground state)
    
    nothing
end

# ╔═╡ energy_curve_plot
let
    p = plot(scan_distances, scan_energies,
        xlabel="Bond Length (Å)",
        ylabel="Energy (Ha)",
        title="H₂ Potential Energy Curve - $(basis_set_name)",
        marker=:circle,
        markersize=6,
        linewidth=2,
        label="HF/$basis_set_name",
        grid=true)
    
    # Mark current point
    scatter!(p, [bond_length_angstrom], [results.final_energy],
        marker=:star,
        markersize=12,
        color=:red,
        label="Current")
    
    # Mark calculated minimum
    scatter!(p, [min_r], [min_e],
        marker=:diamond,
        markersize=10,
        color=:green,
        label="Calculated min")
    
    if show_reference
        # Show experimental bond length as vertical line
        vline!(p, [exp_bond_length],
            linestyle=:dash,
            color=:orange,
            linewidth=2,
            label="Exp. bond length")
    end
    
    annotate!(p, min_r, min_e + 0.02, 
        text(@sprintf("%.3f Å", min_r), 10, :green))
    
    p
end

# ╔═╡ orbital_energies_cell
if results.orbital_energies !== nothing
    let
        # Plot orbital energy level diagram
        ε = results.orbital_energies
        n_occ = length(results.atoms) ÷ 2
        
        p = plot(legend=false, xlim=(0, 3), ylim=(minimum(ε)-0.5, maximum(ε)+0.5),
                xlabel="", ylabel="Energy (Ha)", 
                title="Molecular Orbital Energy Levels",
                xticks=false)
        
        for (i, e) in enumerate(ε)
            color = i <= n_occ ? :blue : :red
            label_text = i <= n_occ ? "occupied" : "virtual"
            plot!(p, [0.8, 2.2], [e, e], linewidth=3, color=color)
            annotate!(p, 2.4, e, text(@sprintf("ε%d = %.3f", i, e), 9, :left))
        end
        
        # Add legend manually
        hline!(p, [minimum(ε)-1], label="Occupied", color=:blue, linewidth=3)
        hline!(p, [minimum(ε)-1], label="Virtual", color=:red, linewidth=3)
        
        p
    end
else
    md"*Orbital energies not available*"
end

# ╔═╡ comparison_cell
md"""
## Basis Set Comparison

Compare results across different basis sets at the current geometry:
"""

# ╔═╡ basis_comparison
let
    basis_sets = ["STO-3G", "cc-pVDZ", "cc-pVTZ"]
    comparison_results = []
    
    coords = [0.0 0.0 0.0; 0.0 0.0 bond_length_bohr]
    
    for basis in basis_sets
        try
            result = run_hf_from_coords(coords, basis, verbose=false, max_iter=50)
            push!(comparison_results, (
                basis = basis,
                n_basis = result.n_basis,
                energy = result.final_energy,
                iterations = result.iterations,
                converged = result.converged
            ))
        catch e
            push!(comparison_results, (
                basis = basis,
                n_basis = "—",
                energy = NaN,
                iterations = "—",
                converged = false
            ))
        end
    end
    
    # Create comparison table - build each row separately to avoid string escaping issues
    table_rows = String[]
    for r in comparison_results
        energy_str = r.converged ? @sprintf("%.6f", r.energy) : "—"
        converged_str = r.converged ? "✅" : "❌"
        row = "| $(r.basis) | $(r.n_basis) | $(energy_str) | $(r.iterations) | $(converged_str) |"
        push!(table_rows, row)
    end
    
    table_header = """
    | Basis Set | # Functions | Final Energy (Ha) | Iterations | Converged |
    |-----------|-------------|-------------------|------------|-----------|
    """
    
    table_body = join(table_rows, "\n")
    footer_text = "\n\n*At bond length: $(round(bond_length_angstrom, digits=3)) Å*"
    
    Markdown.parse(table_header * table_body * footer_text)
end

# ╔═╡ debug_toggle_cell
md"""
---
## Advanced Options

**Show Debug Info:** $(@bind show_debug CheckBox(default=false))

*Enable to see all intermediate matrices and detailed SCF iteration data*
"""

# ╔═╡ debug_info_cell
if show_debug
    md"""
    ## Debug Information
    
    ### Overlap Matrix S:
    $(results.S_matrix)
    
    ### Kinetic Energy Matrix T:
    $(results.T_matrix)
    
    ### Nuclear Attraction Matrix V_nuc:
    $(results.V_nuc_matrix)
    
    ### Core Hamiltonian H_core = T + V_nuc:
    $(results.H_core)
    
    ### Final Fock Matrix F:
    $(results.F_matrix)
    
    ### Orbital Coefficients C:
    $(results.coefficients)
    
    ### Energy at each iteration:
    $(join([@sprintf("Iter %d: %.6f Ha", i, e) for (i, e) in enumerate(results.energy_history)], "\n\n"))
    """
else
    md""
end

# ╔═╡ footer_cell
md"""
---
### About This Notebook

This interactive notebook demonstrates restricted Hartree-Fock for H₂. Try adjusting:
- **Bond length**: See how energy changes with geometry (minimum around 0.74 Å)
- **Basis set**: Larger basis sets give more accurate results but take longer
- **Scan points**: More points give a smoother potential energy curve

The overlap matrix shows how basis functions overlap in space. Diagonal elements are always 1 (perfect self-overlap), while off-diagonal elements decrease as atoms move apart.

**Implementation:** Julia with BasisSets.jl 
**Author:** Keiran Rowell with prompting of Claude Sonnet 4.5 on written Julia code

**Source Code:** 
- `HF_modular.jl` - Main SCF algorithm
- `integrals.jl` - Integral calculations
- Edit either file and this notebook automatically reloads! ⚡

**Experimental Reference:**
- H₂ equilibrium bond length: 0.74144 Å
- Exact ground state energy: -1.1744 Ha (full CI limit)
"""

# ╔═╡ Cell order:
# ╟─title_cell
# ╟─package_cell
# ╟─parameters_cell
# ╟─include_core
# ╠═computation_cell
# ╟─results_cell
# ╟─overlap_matrix_plot
# ╟─convergence_plot
# ╟─energy_curve_cell
# ╠═energy_scan
# ╟─energy_curve_plot
# ╟─orbital_energies_cell
# ╟─comparison_cell
# ╟─basis_comparison
# ╟─debug_toggle_cell
# ╟─debug_info_cell
# ╟─footer_cell
