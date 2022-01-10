
module Plot

using RecipesBase: RecipesBase, @recipe, @series
using LinearAlgebra: eigvals, diag

using Unitful: @u_str
using UnitfulRecipes: UnitfulRecipes
using UnitfulAtomic: UnitfulAtomic

using ..NQCModels: potential, derivative, nstates
using ..AdiabaticModels: AdiabaticModel
using ..DiabaticModels: DiabaticModel

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

@recipe function f(x, model::DiabaticModel; adiabats=true, diabats=true, coupling=false)
    eigs = zeros(length(x), nstates(model))
    diabatic = zeros(length(x), nstates(model))
    couplings = zeros(length(x), nstates(model), nstates(model))
    for i=1:length(x)
        V = potential(model, hcat(x[i]))
        eigs[i,:] .= eigvals(V)
        diabatic[i,:] .= diag(V)
        couplings[i,:,:] .= V
    end

    xguide := "r"
    yguide := "V(r)"

    if adiabats
        for i=1:nstates(model)
            @series begin
                linecolor := :black
                label := i==1 ? "Adiabatic" : ""
                x .* u"bohr", eigs[:,i] .* u"hartree"
            end
        end
    end

    if diabats
        for i=1:nstates(model)
            @series begin
                linecolor := "#FF1F5B"
                label := i==1 ? "Diabatic" : ""
                x .* u"bohr", diabatic[:,i] .* u"hartree"
            end
        end
    end

    if coupling
        for i=1:nstates(model)
            for j=i+1:nstates(model)
                @series begin
                    linecolor := "#009ADE"
                    label := (i == 1 && j == 2) ? "Diabatic coupling" : ""
                    x .* u"bohr", couplings[:,i,j] .* u"hartree"
                end
            end
        end
    end
end

end # module
