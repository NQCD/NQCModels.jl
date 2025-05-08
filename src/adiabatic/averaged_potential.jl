
struct AveragedPotential{M,T} <: AdiabaticModel
    models::M
    tmp_derivative::Matrix{T}
end

AveragedPotential(models, r) = AveragedPotential(models, zero(r))

NQCModels.ndofs(model::AveragedPotential) = NQCModels.ndofs(model.models[1])

function NQCModels.potential(model::AveragedPotential, r::AbstractMatrix)
    V = zero(eltype(r))
    for m in model.models
        V += NQCModels.potential(m, r)
    end
    return V / length(model.models)
end

function NQCModels.potential!(model::AveragedPotential, V::Real, r::AbstractMatrix)
    V = zero(eltype(r))
    for m in model.models
        V += NQCModels.potential(m, r)
    end
    V / length(model.models)
end

function NQCModels.derivative!(model::AveragedPotential, D::AbstractMatrix, r::AbstractMatrix)
    fill!(D, zero(eltype(D)))
    for m in model.models
        NQCModels.derivative!(m, model.tmp_derivative, r)
        D .+= model.tmp_derivative
    end
    D ./= length(model.models)
    return D
end
