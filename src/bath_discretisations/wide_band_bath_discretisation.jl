
abstract type WideBandBathDiscretisation end
NQCModels.nstates(bath::WideBandBathDiscretisation) = length(bath.bathstates)

function fillbathstates!(out::Hermitian, bath::WideBandBathDiscretisation)
    diagonal = view(out, diagind(out)[2:end])
    copy!(diagonal, bath.bathstates)
end

function fillbathcoupling!(out::Hermitian, coupling::Real, bath::WideBandBathDiscretisation, couplings_rescale::Real=1.0)
    first_column = @view out.data[2:end, 1]
    setcoupling!(first_column, bath.bathcoupling, coupling, couplings_rescale)
    first_row = @view out.data[1, 2:end]
    copy!(first_row, first_column)
end

function setcoupling!(out::AbstractVector, bathcoupling::AbstractVector, coupling::Real, couplings_rescale::Real=1.0)
    out .= bathcoupling .* coupling .* couplings_rescale  # bath's states coupling (constant) * Hybridization coupling component V_k(r)
end

function setcoupling!(out::AbstractVector, bathcoupling::Real, coupling::Real, couplings_rescale::Real=1.0)
    fill!(out, bathcoupling * coupling * couplings_rescale)
end
