
"""
    Free()

Zero external potential everywhere. Useful for modelling free particles.

```jldoctest model
julia> model, R = Free(3), rand(3, 10);

julia> potential(model, R)
0

julia> derivative(model, R)
3×10 Matrix{Float64}:
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
```
"""
struct Free <: AdiabaticModel
    dofs::Int
end

Free() = Free(1)

NonadiabaticModels.ndofs(free::Free) = free.dofs

NonadiabaticModels.potential(::Free, ::AbstractMatrix) = 0
NonadiabaticModels.derivative!(::Free, out::AbstractMatrix, ::AbstractMatrix) = out .= 0
