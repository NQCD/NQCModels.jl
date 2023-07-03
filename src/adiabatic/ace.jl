
using NQCBase: au_to_ang, eV_to_au, eV_per_ang_to_au
# using .JuLIP: JuLIP
using .ACE1: ACE1
export ACE1Model

"""
    struct ACE1Model{T} <: AdiabaticModel

Model for interfacing with JuLIP potentials.
"""
struct ACE1Model{T,M} <: AdiabaticModel
    atoms::JuLIP.Atoms{T}
    model::M
end

NQCModels.ndofs(::ACE1Model) = 3

mat(V::AbstractVector{SVec{N,T}}) where {N,T} =
      reshape( reinterpret(T, V), (N, length(V)) )

function NQCModels.potential(model::ACE1Model, R::AbstractMatrix)
    JuLIP.set_positions!(model.atoms, au_to_ang.(R))
    return eV_to_au(ACE1.energy(model.model, model.atoms))
end

function NQCModels.derivative!(model::ACE1Model, D::AbstractMatrix, R::AbstractMatrix)
    JuLIP.set_positions!(model.atoms, au_to_ang.(R))
    force = forces(model.model, model.atoms)
    D .= -eV_per_ang_to_au.(mat(force))
    return D
end
