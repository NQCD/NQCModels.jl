using LinearAlgebra
using RecipesBase
using Unitful, UnitfulRecipes

@recipe function f(x, model::AdiabaticModel)
    V = zeros(size(x))
    D = zeros(size(x))
    for i=1:length(x)
        V[i] = potential(model, hcat(x[i]))
        D[i] = derivative(model, hcat(x[i]))[1]
    end

    xguide --> "r"

    @series begin
        label := "V(r)"
        x .* u"bohr", V .* u"hartree"
    end

    @series begin
        label := "dV(r)dr"
        x .* u"bohr", D .* u"hartree / bohr"
    end
end

@recipe function f(x, model::DiabaticModel; adiabats=true, diabats=true)
    eigs = zeros(length(x), model.n_states)
    diabatic = zeros(length(x), model.n_states)
    for i=1:length(x)
        V = potential(model, hcat(x[i]))
        eigs[i,:] .= eigvals(V)
        diabatic[i,:] .= diag(V)
    end

    legend --> false
    xguide --> "r"
    yguide --> "V(r)"

    if adiabats
        @series begin
            linecolor := :black
            x .* u"bohr", eigs .* u"hartree"
        end
    end

    if diabats
        @series begin
            linecolor := :red
            x .* u"bohr", diabatic .* u"hartree"
        end
    end
end
