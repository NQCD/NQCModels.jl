using PythonCall
using Unitful: @u_str
using UnitfulAtomic: austrip

"""
    ASEFrictionProvider{A} <: ElectronicFrictionProvider

Obtain the electronic friction from an ASE calculator that implements `get_friction_tensor`.
Assumes that the units of friction are "eV/Å/Å".
Construct by passing the ase atoms object with the calculator already attached.
"""
struct ASEFrictionProvider{A} <: TensorialFriction
    atoms::A
end

NQCModels.ndofs(::ASEFrictionProvider) = 3

function friction!(model::ASEFrictionProvider, F::AbstractMatrix, R::AbstractMatrix)
    set_coordinates!(model, R)
    F .= pyconvert(Matrix{eltype(F)}, model.atoms.get_friction_tensor()) # Not transposing since the EFT must be symmetric
    @. F = austrip(F * u"eV/Å/Å")
end

function set_coordinates!(model::ASEFrictionProvider, R)
    model.atoms.set_positions(ustrip.(auconvert.(u"Å", R')))
end
