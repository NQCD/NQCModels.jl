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

"""
    discrete_bath{T} <: DiscreteBath

Stuct containing information for the choice of discretisation for the quantum bath, as well as the density of state representation - given by the `bathfucntion`.


At present, the types of `bathfunction` available include:


- `widebandbath(discretisation::BathDiscretisationScheme)`
- `lorentzianbath(discretisation::BathDiscretisationScheme)`

"""
struct discrete_bath{T} <: DiscreteBath
    bathstates::Vector{T} # Vector of Floats
    bathcoupling::Vector{T} # Vector of Floats
    bathfunction::Vector{T} # Vector of Floats
    bathtype::Symbol
    discretisationtype::Symbol 
end

"""
    discrete_bath(discretisation::BathDiscretisationScheme, bathfn::BathFunction=widebandbath(discretisation))

Returns the struct with details of the discrete quantum bath. 
    
Primary input is the choice of `BathDiscretisationScheme`, with optional keyword argument `bathfn` which dictates if the quantum bath is considered to be in the wideband limit (constant density of states) or given by an analytical function such as a lorentzian.
"""
function discrete_bath(discretisation::BathDiscretisationScheme, bathfn::BathFunction=widebandbath(discretisation))
    (; bathstates, bathcoupling, discretisationtype) = discretisation
    (; bathfunction, bathtype) = bathfn
    return discrete_bath(bathstates, bathcoupling, bathfunction, bathtype, discretisationtype)
end

function fillbathstates!(out::Hermitian, bath::DiscreteBath)
    diagonal = view(out, diagind(out)[2:end])
    copy!(diagonal, bath.bathstates)
end

function fillbathcoupling!(out::Hermitian, coupling::Real, bath::DiscreteBath, couplings_rescale::Real=1.0)
    first_row = @view out.data[1, 2:end] 
    setcoupling!(first_row, bath.bathcoupling, coupling, bath.bathfunction, couplings_rescale)

    return nothing
end

function setcoupling!(out::AbstractVector, bathcoupling::AbstractVector, coupling::Real, bathfunction::AbstractVector, couplings_rescale::Real=1.0)
    @inbounds for i in eachindex(out)
        out[i] = bathcoupling[i] * coupling * bathfunction[i] * couplings_rescale
    end
end

export discrete_bath
export fillbathstates!
export fillbathcoupling!
export setcoupling!

end