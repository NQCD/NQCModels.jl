module BathDiscretisations
using ..NQCModels: NQCModels
using FastGaussQuadrature: gausslegendre
using LinearAlgebra

abstract type BathDiscretisationScheme end
NQCModels.nstates(bath::BathDiscretisationScheme) = length(bath.bathstates) # unsure about this yet

abstract type BathFunction end

include("wide_band_bath_discretisation.jl")
export WideBandBathDiscretisation
export fillbathstates!
export fillbathcoupling!
export setcoupling!
export widebandbath

include("lorentzian_bath.jl")
export lorentzianbath

include("trapezoidal_rule.jl")
export TrapezoidalRule
export GapTrapezoidalRule

include("shenvi_gauss_legendre.jl")
export ShenviGaussLegendre
export ReferenceGaussLegendre

include("full_gauss_legendre.jl")
export FullGaussLegendre
export GapGaussLegendre

include("windowed_trapezoidal_rule.jl")
export WindowedTrapezoidalRule


abstract type DiscreteBath end

struct discrete_bath{B,T,S} <: DiscreteBath
    bathstates::Union{B, Vector{T}} # StepLen or Vector of Floats
    bathcoupling::Union{T, Vector{T}} # Float or Vector of Floats
    bathfunction::Union{T, Vector{T}} # Float or Vector of Floats
    # discretisation_scheme::S # Symbol - only way I can think of for this to work is to have the bathdiscretisationscheme contain its own name as a field
end

function discrete_bath(discretisation::BathDiscretisationScheme, bathfn::BathFunction)
    (; bathstates, bathcoupling) = discretisation
    bathfunction = bathfn(discretisation)
    return discrete_bath(bathstates, bathcoupling, bathfunction)
end

end