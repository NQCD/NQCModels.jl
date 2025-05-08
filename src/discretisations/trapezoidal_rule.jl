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
