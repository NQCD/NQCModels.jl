using Test
using NQCBase
using NQCModels
using LinearAlgebra
using FiniteDiff
using SafeTestsets

@time @safetestset "Wide band bath discretisations" begin include("wide_band_bath_discretisations.jl") end

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

@testset "Potential abstraction" begin
    struct TestModel <: NQCModels.Model end

    NQCModels.ndofs(::TestModel) = 3
    @test_throws MethodError potential(TestModel(), rand(3,1))

    NQCModels.ndofs(::TestModel) = 1
    NQCModels.potential(::TestModel, ::Real) = 1
    NQCModels.potential(::TestModel, ::AbstractVector) = 2
    @test potential(TestModel(), rand(1,1)) == 1
    @test potential(TestModel(), rand(1,2)) == 2
end

@testset "Plot" begin
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
    @test test_model(AveragedPotential((Harmonic(), Harmonic()), zeros(1, 10)), 10)
    @test test_model(BosonBath(OhmicSpectralDensity(2.5, 0.1), 10), 10)
    @test test_model(DarlingHollowayElbow(), 2)
    @test test_model(Morse(), 1)
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
    @test test_model(MiaoSubotnik(), 1)
    @test test_model(AnanthModelOne(), 1)
    @test test_model(AnanthModelTwo(), 1)
    @test test_model(ErpenbeckThoss(Γ=2.0), 1)
    @test test_model(WideBandBath(ErpenbeckThoss(Γ=2.0); step=0.1, bandmin=-1.0, bandmax=1.0), 1)
    @test test_model(WideBandBath(GatesHollowayElbow(); step=0.1, bandmin=-1.0, bandmax=1.0), 2)
    @test test_model(AndersonHolstein(ErpenbeckThoss(Γ=2.0), TrapezoidalRule(10, -1, 1)), 1)
    @test test_model(AndersonHolstein(ErpenbeckThoss(Γ=2.0), ShenviGaussLegendre(10, -1, 1)), 1)
    @test test_model(AndersonHolstein(GatesHollowayElbow(), ShenviGaussLegendre(10, -1, 1)), 2)
end

@testset "FrictionModels" begin
    @test test_model(CompositeFrictionModel(Free(2), ConstantFriction(2, 1)), 3)
    @test test_model(CompositeFrictionModel(Free(3), RandomFriction(3)), 3)
end

@testset "JuLIP" begin
    using JuLIP: JuLIP
    at = JuLIP.bulk(:Si, cubic=true)
    deleteat!(at, 1)
    JuLIP.set_calculator!(at, JuLIP.StillingerWeber())
    model = AdiabaticModels.JuLIPModel(at)
    @test test_model(model, length(at))
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
