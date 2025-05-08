module BathDiscretisations

using FastGaussQuadrature: gausslegendre

include("wide_band_bath_discretisation.jl")
export WideBandBathDiscretisation
export fillbathstates!
export fillbathcoupling!
export setcoupling!

include("trapezoidal_rule.jl")
export TrapezoidalRule
include("shenvi_gauss_legendre.jl")
export ShenviGaussLegendre
export ReferenceGaussLegendre
include("full_gauss_legendre.jl")
export FullGaussLegendre

end