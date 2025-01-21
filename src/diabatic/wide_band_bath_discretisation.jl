using FastGaussQuadrature: gausslegendre

abstract type WideBandBathDiscretisation end
NQCModels.nstates(bath::WideBandBathDiscretisation) = length(bath.bathstates)

function fillbathstates!(out::Hermitian, bath::WideBandBathDiscretisation)
    diagonal = view(out, diagind(out)[2:end])
    copy!(diagonal, bath.bathstates)
end

function fillbathcoupling!(out::Hermitian, coupling::Real, bath::WideBandBathDiscretisation)
    first_column = @view out.data[2:end, 1]
    setcoupling!(first_column, bath.bathcoupling, coupling)
    first_row = @view out.data[1, 2:end]
    copy!(first_row, first_column)
end

function setcoupling!(out::AbstractVector, bathcoupling::AbstractVector, coupling::Real)
    out .= bathcoupling .* coupling
end

function setcoupling!(out::AbstractVector, bathcoupling::Real, coupling::Real)
    fill!(out, bathcoupling * coupling)
end

"""
    TrapezoidalRule{B,T} <: WideBandBathDiscretisation

Discretise wide band continuum using trapezoidal rule.
Leads to evenly spaced states and constant coupling.
"""
struct TrapezoidalRule{B,T} <: WideBandBathDiscretisation
    bathstates::B
    bathcoupling::T
end

function TrapezoidalRule(M, bandmin, bandmax)
    ΔE = bandmax - bandmin
    bathstates = range(bandmin, bandmax, length=M)
    coupling = sqrt(ΔE / M)
    return TrapezoidalRule(bathstates, coupling)
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
    
    M_window = floor(Int, M*densityratio)
    M_sparse = floor(Int, (M - (M*densityratio))/2)
    

    # densely distributed trapezoidal rule discretised bath states within energy window
    ΔE_window = windmax - windmin
    bstates_window = collect(range(windmin, windmax, length=M_window))
    bcoupling_window = fill!(copy(bstates_window), sqrt(ΔE_window / M_window))

    # Sparsely distributed trapezoidal rule discretised bath states
    ΔE_sparse1 = windmin - bandmin # Energy range for first sparsely distributed state region
    bstates_sparse1 = collect(range(bandmin, windmin, length=M_sparse+1))[1:end-1] # slicing is implemented to rectify energy degeneracy at window bounds
    bcoupling_sparse1 = fill!(copy(bstates_sparse1), sqrt(ΔE_sparse1 / M_sparse+1))


    ΔE_sparse2 = bandmax - windmax # Energy Range for second sparsely distributed state region
    bstates_sparse2 = collect(range(windmax, bandmax, length=M_sparse+1))[2:end]
    bcoupling_sparse2 = fill!(copy(bstates_sparse2), sqrt(ΔE_sparse2 / M_sparse+1))
    


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
