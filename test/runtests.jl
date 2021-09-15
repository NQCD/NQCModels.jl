using Test
using NonadiabaticDynamicsBase
using NonadiabaticModels
using LinearAlgebra
using FiniteDiff

function finite_difference_gradient(model::NonadiabaticModels.AdiabaticModels.AdiabaticModel, R)
    f(x) = potential(model, x)
    FiniteDiff.finite_difference_gradient(f, R)
end

function finite_difference_gradient(model::NonadiabaticModels.DiabaticModels.DiabaticModel, R)
    f(x, i, j) = potential(model, x)[i,j]
    grad = [Hermitian(zeros(nstates(model), nstates(model))) for i in CartesianIndices(R)]
    for k in eachindex(R)
        for i=1:nstates(model)
            for j=1:nstates(model)
                grad[k].data[i,j] = FiniteDiff.finite_difference_gradient(x->f(x,i,j), R)[k]
            end
        end
    end
    grad
end

function test_model(model::NonadiabaticModels.Model, atoms; rtol=1e-5)
    R = rand(ndofs(model), atoms)
    D = derivative(model, R)
    finite_diff = finite_difference_gradient(model, R)
    return isapprox(finite_diff, D, rtol=rtol)
end

function test_model(model::NonadiabaticModels.FrictionModels.AdiabaticFrictionModel, atoms)
    R = rand(ndofs(model), atoms)
    D = derivative(model, R)
    friction(model, R)
    return finite_difference_gradient(model, R) ≈ D
end

@testset "plot" begin
    using Plots
    plot(-10:0.1:10, Harmonic())
    plot(-10:0.1:10, DoubleWell())
end

@testset "DiatomicHarmonic" begin
    model = DiatomicHarmonic()
    @test test_model(model, 2)

    R = [0 0; 0 0; 1 0]
    @test potential(model, R) ≈ 0
    R = [sqrt(3) 0; sqrt(3) 0; sqrt(3) 0]
    @test potential(model, R) ≈ 2
end

@testset "AdiabaticModels" begin
    @test test_model(Harmonic(), 10)
    @test test_model(Free(), 10)
    @test test_model(BosonBath(OhmicSpectralDensity(2.5, 0.1), 10), 10)
    @test test_model(DarlingHollowayElbow(), 2)
end

@testset "DiabaticModels" begin
    @test test_model(DoubleWell(), 1)
    @test test_model(TullyModelOne(), 1)
    @test test_model(TullyModelTwo(), 1)
    @test test_model(TullyModelThree(), 1)
    @test test_model(Scattering1D(), 1)
    @test test_model(ThreeStateMorse(), 1)
    @test test_model(SpinBoson(DebyeSpectralDensity(0.25, 0.5), 10, 1.0, 1.0), 10)
    @test test_model(OuyangModelOne(), 1)
    @test test_model(GatesHollowayElbow(), 2)
    # test_model(Subotnik_A(), 1, 1) broken
    @test test_model(MiaoSubotnik(), 1)
end

@testset "FrictionModels" begin
    @test test_model(ConstantFriction(Free(), 1), 3)
    @test test_model(RandomFriction(Free()), 3)
end

@testset "JuLIP" begin
    import JuLIP
    atoms = Atoms([:H, :H])
    vecs = [10 0 0; 0 10 0; 0 0 10]
    model = AdiabaticModels.JuLIPModel(atoms, PeriodicCell(vecs), JuLIP.StillingerWeber())
    @test_broken test_model(model, 2)
end

@testset "ASE" begin
    using PyCall

    ase = pyimport("ase")

    h2 = ase.Atoms("H2", [(0, 0, 0), (0, 0, 0.74)])
    h2.center(vacuum=2.5)

    @testset "EMT" begin
        emt = pyimport("ase.calculators.emt")
        h2.calc = emt.EMT()
        model = AdiabaticASEModel(h2)
        @test test_model(model, 2)
    end

    @testset "GPAW" begin
        gpaw = pyimport("gpaw")
        h2.calc = gpaw.GPAW(xc="PBE", mode=gpaw.PW(300), txt="h2.txt")
        model = AdiabaticASEModel(h2)
        @test test_model(model, 2, rtol=1e-3)
    end

end
