using FastGaussQuadrature: gausslegendre

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

"""
    TrapezoidalRule{B,T} <: WideBandBathDiscretisation

Discretise wide band continuum using trapezoidal rule.
Leads to evenly spaced states and constant coupling.
"""
struct TrapezoidalRule{B,T} <: WideBandBathDiscretisation
    bathstates::B   # ϵ
    bathcoupling::T # V(ϵ,x̃) 
end

function TrapezoidalRule(M, bandmin, bandmax)
    ΔE = bandmax - bandmin
    bathstates = range(bandmin, bandmax, length=M)
    bathcoupling = sqrt(ΔE / M)
    return TrapezoidalRule(bathstates, bathcoupling)
end

# ------------------------------------------------------------------------------------------------ #
#                                     New discretisation method                                    #
# ------------------------------------------------------------------------------------------------ #

"""
    WindowedTrapezoidalRule{B,T}

Discretise wide band continuum using the trapezoidal rule with a region of higher state density
surrounding the Fermi Energy. The position of the energy window containing more densely packed states
is defined by energy values and the increase in density by a proportion. 
"""

struct WindowedTrapezoidalRule{T} <: WideBandBathDiscretisation
    bathstates::Vector{T} # due to the non-even distribution need to represent as vectors
    bathcoupling::Vector{T} 
end

function WindowedTrapezoidalRule(M, bandmin, bandmax, windmin, windmax; densityratio=0.50)
    (M - (M*densityratio)) % 2 == 0 || throw(error("For the provided arguments, the number of states not in the windowed region is $(M - (M*densityratio)). This value must be even for a symmetric discretisation.")) # constraint enforced such that the a densityratio=0.5 can be utilized and return an integer number of states.
    abs(bandmin) > abs(windmin) || throw(error("Requested window minimum energy lies outside of energy range."))
    abs(bandmax) > abs(windmax) || throw(error("Requested window maximum energy lies outside of energy range."))
    
    M_sparse = floor(Int, (M - (M*densityratio))/2)
    M_window = floor(Int, M*densityratio)

    # Sparsely distributed trapezoidal rule discretised bath states
    ΔE_sparse1 = windmin - bandmin # Energy range for first sparsely distributed state region
    bstates_sparse1 = collect(range(bandmin, windmin, length=M_sparse))
    bcoupling_sparse1 = fill!(copy(bstates_sparse1), sqrt(ΔE_sparse1 / M_sparse))

    ΔE_sparse2 = bandmax - windmax # Energy Range for second sparsely distributed state region
    bstates_sparse2 = collect(range(windmax, bandmax, length=M_sparse))
    bcoupling_sparse2 = fill!(copy(bstates_sparse2), sqrt(ΔE_sparse2 / M_sparse))

    # densely distributed trapezoidal rule discretised bath states within energy window
    ΔE_window = windmax - windmin
    bstates_window = collect(range(windmin, windmax, length=M_window))
    bcoupling_window = fill!(copy(bstates_window), sqrt(ΔE_window / M_window))

    bathstates = [bstates_sparse1; bstates_window; bstates_sparse2] # concatenates arrays vertically (along axis = 1)    
    bathcoupling = [bcoupling_sparse1; bcoupling_window; bcoupling_sparse2]

    return WindowedTrapezoidalRule(bathstates, bathcoupling)
end

# ------------------------------------------------------------------------------------------------ #

"""
    ShenviGaussLegendre{T}

Defined as described by Shenvi et al. in J. Chem. Phys. 130, 174107 (2009).
The position of the negative sign for the state energy level has been moved to ensure the states are sorted from lowest to highest.
"""
struct ShenviGaussLegendre{T} <: WideBandBathDiscretisation
    bathstates::Vector{T}
    bathcoupling::Vector{T}
end

function ShenviGaussLegendre(M, bandmin, bandmax)
    M % 2 == 0 || throw(error("The number of states `M` must be even."))
    knots, weights = gausslegendre(div(M, 2))
    centre = (bandmax + bandmin) / 2

    bathstates = zeros(M)
    for i in eachindex(knots)
        bathstates[i] = (centre - bandmin)/2 * knots[i] + (bandmin + centre) / 2
    end
    for i in eachindex(knots)
        bathstates[i+length(knots)] = (bandmax - centre)/2 * knots[i] + (bandmax + centre) / 2
    end

    bathcoupling = zeros(M)
    for (i, w) in enumerate(weights)
        bathcoupling[i] = sqrt((centre - bandmin)/2  * w)
    end
    for (i, w) in enumerate(weights)
        bathcoupling[i+length(weights)] = sqrt((bandmax - centre)/2  * w)
    end

    return ShenviGaussLegendre(bathstates, bathcoupling)
end

"""
    ReferenceGaussLegendre{T}

Implementation translated from Fortran code used for simulations of Shenvi et al. in J. Chem. Phys. 130, 174107 (2009).
Two differences from ShenviGaussLegendre:
- Position of minus sign in energy levels has been corrected.
- Division by sqrt(ΔE) in the coupling. 
"""
struct ReferenceGaussLegendre{T} <: WideBandBathDiscretisation
    bathstates::Vector{T}
    bathcoupling::Vector{T}
end

function ReferenceGaussLegendre(M, bandmin, bandmax)
    M % 2 == 0 || throw(error("The number of states `M` must be even."))
    @warn "(ReferenceGaussLegendre) The division by sqrt(ΔE) in the coupling is incorrect. Use ShenviGaussLegendre instead."
    knots, weights = gausslegendre(div(M, 2))
    ΔE = bandmax - bandmin

    bathstates = zeros(M)
    for i in eachindex(knots)
        bathstates[i] = ΔE/2 * (-1/2 + knots[i]/2)
    end
    for i in eachindex(knots)
        bathstates[i+length(knots)] = ΔE/2 * (1/2 + knots[i]/2)
    end

    bathcoupling = zeros(M)
    for (i, w) in zip(eachindex(bathcoupling), Iterators.cycle(weights))
        bathcoupling[i] = sqrt(w) / 2
    end

    return ReferenceGaussLegendre(bathstates, bathcoupling)
end

"""
    FullGaussLegendre{T} <: WideBandBathDiscretisation

Use Gauss-Legendre quadrature to discretise the continuum across the entire band width.
This is similar to the ShenviGaussLegendre except that splits the continuum at the Fermi level into two halves.
"""
struct FullGaussLegendre{T} <: WideBandBathDiscretisation
    bathstates::Vector{T}
    bathcoupling::Vector{T}
end

function FullGaussLegendre(M, bandmin, bandmax)

    knots, weights = gausslegendre(M)
    ΔE = bandmax - bandmin

    bathstates = zeros(M)
    for i in eachindex(knots, bathstates)
        bathstates[i] = ΔE/2 * knots[i]
    end

    bathcoupling = zeros(M)
    for i in eachindex(bathcoupling, weights)
        bathcoupling[i] = sqrt(weights[i] * ΔE/2)
    end

    return FullGaussLegendre(bathstates, bathcoupling)
end

"""
The **`GapGaussLegendre`** struct, a subtype of `WideBandBathDiscretisation`, models a wide-band limit bath with a central energy gap using Gauss–Legendre quadrature for discretization.

## Fields

- **`bathstates::Vector{T}`**: The discretized energy levels of the bath.
- **`bathcoupling::Vector{T}`**: The coupling strengths to the bath states, derived from Gauss–Legendre weights.

## Constructor

```julia
GapGaussLegendre(M, bandmin, bandmax, gapwidth)
```

This constructor validates input arguments and initializes a `GapGaussLegendre` object using Gauss–Legendre quadrature for state placement and coupling computation.

## Arguments

- `M::Int`: The total number of bath states; must be even.
- `bandmin::Real`: The minimum energy of the wide band.
- `bandmax::Real`: The maximum energy of the wide band.
- `gapwidth::Real`: The width of the central energy gap. Must be non-negative and smaller than `bandmax - bandmin`.

## Details

The energy band is split into two symmetric regions around the gap, and `M/2` states are placed in each region using Gauss–Legendre nodes mapped to the corresponding subinterval. The gap itself is avoided.

The `bathcoupling` vector is computed from the square roots of the Gauss–Legendre weights, scaled by the effective half-bandwidth. This ensures accurate integration over the energy band while excluding the gap.
"""
struct GapGaussLegendre{T} <: WideBandBathDiscretisation
    bathstates::Vector{T}
    bathcoupling::Vector{T}
end

function GapGaussLegendre(M, bandmin, bandmax, gapwidth)
    gapwidth < bandmax - bandmin || throw(error("The gap width must be smaller than the band width."))
    gapwidth >= 0 || throw(error("The gap width must be positive."))
    M % 2 == 0 || throw(error("The number of states `M` must be even."))
    knots, weights = gausslegendre(div(M, 2))
    centre = (bandmax + bandmin) / 2
    ΔE = bandmax - bandmin
    bandlengths = (ΔE - gapwidth)/2

    bathstates = zeros(M)
    for i in eachindex(knots)
        bathstates[length(knots)+1-i] = ((ΔE-gapwidth)/2 * (1 + knots[i]) + gapwidth) * (-1/2) + centre
    end
    for i in eachindex(knots)
        bathstates[i+length(knots)] = ((ΔE-gapwidth)/2 * (1 + knots[i]) + gapwidth) * 1/2 + centre
    end

    # coupling is the sqrt() of the weights times the bandlengths [valence band's coupling ; conduction band's coupling]
    bathcoupling = [sqrt.(weights.*bandlengths/2); sqrt.(weights.*bandlengths/2)]

    return GapGaussLegendre(bathstates, bathcoupling)
end

"""
The **`GapTrapezoidalRule`** struct, a subtype of `WideBandBathDiscretisation`, models a wide-band limit bath with an energy gap.

## Fields

- **`bathstates::Vector{T}`**: The discretized energy levels of the bath.
- **`bathcoupling::T`**: The coupling strength between the system and the bath states.

## Constructor

```julia
GapTrapezoidalRule(M, bandmin, bandmax, gapwidth)
```
This constructor validates inputs and initializes a GapTrapezoidalRule object.

## Arguments

- `M::Int`: The total number of bath states; must be even.
- `bandmin::Real`: The minimum energy of the wide band.
- `bandmax::Real`: The maximum energy of the wide band.
- `gapwidth::Real`: The width of the central energy gap. Must be non-negative and smaller than `bandmax - bandmin`.

## Details

If `gapwidth` is zero, `bathstates` are uniformly distributed across the entire band. Otherwise, the band is split, and `M` states are evenly distributed across the two resulting halves, avoiding the gap.

The `bathcoupling` is calculated from the effective bandwidth (`bandmax - bandmin - gapwidth`) and `M`.

"""


struct GapTrapezoidalRule{T} <: WideBandBathDiscretisation
    bathstates::Vector{T}   # ϵ
    bathcoupling::T # V(ϵ,x̃) 
end

function GapTrapezoidalRule(M, bandmin, bandmax, gapwidth)
    gapwidth < bandmax - bandmin || throw(error("The gap width must be smaller than the band width."))
    gapwidth >= 0 || throw(error("The gap width must be positive."))
    M % 2 == 0 || throw(error("The number of states `M` must be even."))
    ΔE = bandmax - bandmin
    centre = (bandmax + bandmin) / 2

    if gapwidth == 0
        bathstates = collect(range(bandmin, bandmax, length=M))
    else
        bathstates_left = range(bandmin, centre - gapwidth/2, length=Int(M/2))
        bathstates_right = range(centre + gapwidth/2, bandmax, length=Int(M/2))
        bathstates = vcat(bathstates_left, bathstates_right)
    end

    bathcoupling = sqrt((ΔE-gapwidth) / M)
    return GapTrapezoidalRule(bathstates, bathcoupling)
end
