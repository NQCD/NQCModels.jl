
using NQCBase: au_to_ang, eV_to_au, eV_per_ang_to_au
using .JuLIP: JuLIP

export JuLIPModel

"""
    struct JuLIPModel{T} <: AdiabaticModel

Model for interfacing with JuLIP potentials.
"""
struct JuLIPModel{T} <: AdiabaticModel
    atoms::JuLIP.Atoms{T}
end

NonadiabaticModels.ndofs(::JuLIPModel) = 3

function NonadiabaticModels.potential(model::JuLIPModel, R::AbstractMatrix)
    JuLIP.set_positions!(model.atoms, au_to_ang.(R))
    return eV_to_au(JuLIP.energy(model.atoms))
end

function NonadiabaticModels.derivative!(model::JuLIPModel, D::AbstractMatrix, R::AbstractMatrix)
    JuLIP.set_positions!(model.atoms, au_to_ang.(R))
    force = JuLIP.forces(model.atoms)
    D .= -eV_per_ang_to_au.(JuLIP.mat(force))
    return D
end
