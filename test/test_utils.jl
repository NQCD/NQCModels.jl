using FiniteDiff
using NQCModels

function finite_difference_gradient(model::NQCModels.AdiabaticModels.AdiabaticModel, R)
    f(x) = potential(model, x)
    FiniteDiff.finite_difference_gradient(f, R)
end

function finite_difference_gradient(model::NQCModels.DiabaticModels.DiabaticModel, R)
    f(x, j, i) = potential(model, x)[j,i]
    grad = [Hermitian(zeros(nstates(model), nstates(model))) for _ in CartesianIndices(R)]
    for i=1:nstates(model)
        for j=1:nstates(model)
            gradient = FiniteDiff.finite_difference_gradient(x->f(x,j,i), R)
            for k in eachindex(R)
                grad[k].data[j,i] = gradient[k]
            end
        end
    end
    grad
end

function test_model(model::NQCModels.Model, atoms; rtol=1e-5)
    R = rand(ndofs(model), atoms)
    D = derivative(model, R)
    finite_diff = finite_difference_gradient(model, R)
    return isapprox(finite_diff, D, rtol=rtol)
end

function test_model(model::NQCModels.FrictionModels.AdiabaticFrictionModel, atoms)
    R = rand(ndofs(model), atoms)
    D = derivative(model, R)
    friction(model, R)
    return finite_difference_gradient(model, R) ≈ D
end
