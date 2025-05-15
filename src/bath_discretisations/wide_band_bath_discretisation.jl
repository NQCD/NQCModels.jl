abstract type WideBandBathDiscretisation end
NQCModels.nstates(bath::WideBandBathDiscretisation) = length(bath.bathstates)

function fillbathstates!(out::Hermitian, bath::WideBandBathDiscretisation)
    diagind(out)[2:end] .= bath.bathstates
end

function fillbathcoupling!(out::Hermitian, coupling::Real, bath::WideBandBathDiscretisation) 
    @. out[2:end, 1] = bath.bathcoupling * coupling
    out[1, 2:end] .= first_column
end

function setcoupling!(out::AbstractVector, bathcoupling::AbstractVector, coupling::Real)
    out .= bathcoupling .* coupling
end

function setcoupling!(out::AbstractVector, bathcoupling::Real, coupling::Real)
    fill!(out, bathcoupling * coupling)
end