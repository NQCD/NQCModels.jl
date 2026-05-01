module BathDiscretisations
using ..NQCModels: NQCModels
using FastGaussQuadrature: gausslegendre
using LinearAlgebra

abstract type BathDiscretisationScheme end
NQCModels.nstates(bath::BathDiscretisationScheme) = length(bath.bathstates) # unsure about this yet

include("wide_band_bath_discretisation.jl")
export WideBandBathDiscretisation
export fillbathstates!
export fillbathcoupling!
export setcoupling!

include("lorentzian_bath.jl")

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

end