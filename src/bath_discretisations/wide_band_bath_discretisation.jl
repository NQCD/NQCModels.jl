abstract type WideBandBathDiscretisation end
NQCModels.nstates(bath::WideBandBathDiscretisation) = length(bath.bathstates)

function fillbathstates!(out::Hermitian, bath::WideBandBathDiscretisation)
    out.data[diagind(out)[2:end]] .= bath.bathstates
end

function fillbathcoupling!(out::Hermitian, coupling::Real, bath::WideBandBathDiscretisation) 
    @. out.data[1, 2:end] = bath.bathcoupling * coupling
end

function setcoupling!(out::AbstractVector, bathcoupling::AbstractVector, coupling::Real)
    out .= bathcoupling .* coupling
end

function setcoupling!(out::AbstractVector, bathcoupling::Real, coupling::Real)
    fill!(out, bathcoupling * coupling)
end