using Unitful, UnitfulAtomic
"""
    lorentzian(energy::Float64, width::Float64)

Lineshape equation for Lorentzian (Cauchy) distribution.
"""
function lorentzian(energy::Float64, width::Float64)
    return width / (width^2 + energy^2)
end

"""
    lorentzian_integral(E_range::Tuple, W)

Analytic form for the indefinite integral for a lorentzian within the energy interval defined as (E_a, E_b).
"""
function lorentzian_integral(E_interval::Tuple, W)
    E_a, E_b = E_interval
    return N/π * (atan(E_b/W) - atan(E_a/W))
end

"""
    lorentzianbath{T} <: BathFunction

Struct containing fields relating a discrete quantum bath with density of states defined by a Lorentizian function.
"""
struct lorentzianbath{T} <: BathFunction
    bathfunction :: Vector{T}
    bathdegeneracy :: Vector{T}
    bathtype :: Symbol
end

"""
    lorentzianbath(discretisation::BathDiscretisationScheme; W=4.5)

Args:
- `discretisation`: chosed discretisation scheme for quantum bath
- `W`: width of lorenzian lineshape
"""
function lorentzianbath(discretisation::BathDiscretisationScheme; W=4.5)
    W_au = austrip(W*u"eV")
    bathfunction = lorentzian.(discretisation.bathstates, W_au)
    bathdegeneracy = lorentzian_integral.(discretisation_energy_intervals(discretisation), W_au)
    bathtype = :lorentzian
    return lorentzianbath(bathfunction, bathdegeneracy, bathtype)
end