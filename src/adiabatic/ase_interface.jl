
"""
    AdiabaticASEModel{A} <: AdiabaticModel

Wrapper for an `ase.Atoms` object that has a calculator attached.
This will synchronise the positions with the `ase` object and handle the unit conversions.

Implements both `potential` and `derivative!`.
"""
struct AdiabaticASEModel{A} <: AdiabaticModel
    atoms::A
end

NonadiabaticModels.ndofs(::AdiabaticASEModel) = 3

function NonadiabaticModels.potential(model::AdiabaticASEModel, R::AbstractMatrix)
    set_coordinates!(model, R)
    V = model.atoms.get_potential_energy()
    return austrip(V * u"eV")
end

function NonadiabaticModels.derivative!(model::AdiabaticASEModel, D::AbstractMatrix, R::AbstractMatrix)
    set_coordinates!(model, R)
    D .= -model.atoms.get_forces()'
    @. D = austrip(D * u"eV/Å")
    return D
end

function set_coordinates!(model::AdiabaticASEModel, R)
    model.atoms.set_positions(ustrip.(auconvert.(u"Å", R')))
end
