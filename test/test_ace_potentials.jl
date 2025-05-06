using Test
using AtomsCalculators
using ACEpotentials
using NQCBase
using NQCModels
using NQCBase: au_to_ang, au_to_eV, eV_to_au, eV_per_ang_to_au, au_to_eV_per_ang, System


n_fixed_atoms = 18
model_path = "ace_potentials_model/model.json"
atoms_path = "ace_potentials_model/h2cu_start.in"

atoms_ase = io.read(atoms_path)
atoms, R, cell = NQCBase.read_extxyz("ace_potentials_model/h2cu_start.in")
R=R[1] # ExtXYZ always returns a Vector, even for length 1. 

ace_model, ace_model_meta = ACEpotentials.load_model(model_path)
model = AdiabaticModels.ACEpotentialsModel(atoms, cell, ace_model) 

@testset "ACEpotentialsModel!" begin
    DoFs = size(R, 1)
    
    # E = au_to_eV(NQCModels.potential(model, xR))
    D = NQCModels.zero_derivative(model, R)
    F = au_to_eV_per_ang.(NQCModels.derivative!(model, D, xR))[n_fixed_atoms*3+1:end] .* -1

    system = System(atoms, R, cell)
    force = AtomsCalculators.forces(system, ace_model)
    F_direct .= austrip.(auconvert.(transpose(reduce(vcat,transpose.(force)))))[n_fixed_atoms*3+1:end]

    @test F ≈ F_direct
end
