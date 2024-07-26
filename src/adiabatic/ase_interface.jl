
"""
    AdiabaticASEModel{A} <: AdiabaticModel

Wrapper for an `ase.Atoms` object that has a calculator attached.
This Model will synchronise the positions with the `ase` object and handle the unit conversions.

Implements both `potential` and `derivative!`.


## Notes on calling Python from Julia

Both PyCall.jl and PythonCall.jl can be used to create the `ase.Atoms` object, but **PythonCall.jl is preferred**.
NQCModels installs a Conda environment containing ase and MACE by default as part of package installation. 


"""
struct AdiabaticASEModel{A} <: AdiabaticModel
    atoms::A
end

NQCModels.ndofs(::AdiabaticASEModel) = 3

function NQCModels.potential(model::AdiabaticASEModel, R::AbstractMatrix)
    set_coordinates!(model, R)
    V = model.atoms.get_potential_energy()
    return austrip(pyconvert(eltype(R),V) * u"eV")
end

function NQCModels.derivative!(model::AdiabaticASEModel, D::AbstractMatrix, R::AbstractMatrix)
    set_coordinates!(model, R)
    D .= -pyconvert(Matrix{eltype(D)}, model.atoms.get_forces())'
    @. D = austrip(D * u"eV/Å")
    return D
end

function set_coordinates!(model::AdiabaticASEModel, R)
    model.atoms.set_positions(ustrip.(auconvert.(u"Å", R')))
end

"""
This module contains methods related to the NQCModels ASE interface that need access to Python types. (e.g. constraint checking)
"""

function NQCModels.mobileatoms(model::AdiabaticASEModel, n::Int)
    return symdiff(1:length(model.atoms), [pyconvert(Vector,constraint.get_indices()) .+ 1 for constraint in model.atoms.constraints]...)
end
