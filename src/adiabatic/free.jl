export Free

"""
    Free()

Zero external potential everywhere. Useful for modelling free particles.

```jldoctest model
julia> model, R = Free(), rand(3, 10);

julia> potential(model, R)
1-element Vector{Float64}:
 0.0

julia> derivative(model, R)
3×10 Matrix{Float64}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
struct Free <: AdiabaticModel end

potential!(::Free, out::AbstractVector, ::AbstractMatrix) = out .= 0
derivative!(::Free, out::AbstractMatrix, ::AbstractMatrix) = out .= 0
