"""
This a building file for the constant DOS but with a Germanium DOS

    This should be a replicate of the :TrapezoidalRule from the wide_band_bath_discretisation.jl
"""

abstract type WideBandBathDiscretisation end

"""
    TrapezoidalRuleGap{B,T} <: WideBandBathDiscretisation

Discretise wide band continuum with a Gap using trapezoidal rule.
Leads to evenly spaced states and constant coupling.
"""
struct TrapezoidalRuleGap{B,T} <: WideBandBathDiscretisation
    bathstates::B   # ϵ
    bathcoupling::T # V(ϵ,x̃) 
end

function TrapezoidalRuleGap(M, bandmin, bandmax, gapmin, gapmax)
    ΔE = bandmax - bandmin
    bathstates = collect(range(bandmin, bandmax, length=M))  # Convert range to array
    partition = ΔE / (M-1) # got confused about here, should it be M-1 or M?

    # gap's index
    # 天花板: ceiling 地板: floor
    天花板 = ceil(Int64,(gapmin - bandmin)/partition + 1)

    地板 = floor(Int64,(gapmax - bandmin)/partition + 1)


    bathstates[天花板:地板] .= 0  # turns states in the gap to 0
    coupling = sqrt(partition)
    return TrapezoidalRuleGap(bathstates, coupling)  # Return the struct with the updated array
end
