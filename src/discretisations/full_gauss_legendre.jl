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