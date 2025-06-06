using Test
using NQCBase
using NQCModels
# using NQCDInterfASE
using LinearAlgebra
using SafeTestsets
using InteractiveUtils

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Composite"
    @testset "CompositeModel" begin
        osc_1 = Harmonic(; ω = 1.0)
        osc_2 = Harmonic(; ω = 2.0)
        composite = CompositeModel(
            Subsystem(osc_1, 1:1),
            Subsystem(osc_2, 2:2)
        )
        positions = [3.45 1.23]
        combined_derivative = hcat(
            NQCModels.derivative(osc_1, positions[:, 1:1]),
            NQCModels.derivative(osc_2, positions[:, 2:2]),
        )
        combined_derivative_values = zero(positions)
        NQCModels.derivative!(composite, combined_derivative_values, positions)
        @test combined_derivative == combined_derivative_values
    end
end

@time @safetestset "Wide band bath discretisations" begin
    include("wide_band_bath_discretisations.jl")
end
@time @safetestset "Anderson Holstein" begin
    include("anderson_holstein.jl")
end
@safetestset "AdiabaticStateSelector" begin
    include("test_adiabatic_state_selector.jl")
end

include("test_utils.jl")

# if GROUP =="All" || GROUP == "ASE"
#     @time @safetestset "ASE with PythonCall.jl" begin
#         include("ase_pythoncall.jl")
#     end
    
#     @time @safetestset "ASE with PyCall.jl" begin
#         include("ase_pycall.jl")
#     end
# end

@testset "Potential abstraction" begin
    struct TestModel <: NQCModels.Model end

    NQCModels.ndofs(::TestModel) = 3
    @test_throws MethodError potential(TestModel(), rand(3, 1))

    NQCModels.ndofs(::TestModel) = 1
    NQCModels.potential(::TestModel, ::Real) = 1
    NQCModels.potential(::TestModel, ::AbstractVector) = 2
    @test potential(TestModel(), rand(1, 1)) == 1
    @test potential(TestModel(), rand(1, 2)) == 2
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

if GROUP == "All" || GROUP == "Classical"
    @testset "ClassicalModels" begin
        @test test_model(Harmonic(), 10)
        @test test_model(Free(), 10)
        @test test_model(AveragedPotential((Harmonic(), Harmonic()), zeros(1, 10)), 10)
        @test test_model(DarlingHollowayElbow(), 2)
        @test test_model(Morse(), 1)
    end
end

if GROUP == "All" || GROUP == "Quantum"
    @testset "QuantumModels" begin
        @test test_model(DoubleWell(), 1)
        @test test_model(TullyModelOne(), 1)
        @test test_model(TullyModelTwo(), 1)
        @test test_model(TullyModelThree(), 1)
        @test test_model(Scattering1D(), 1)
        @test test_model(ThreeStateMorse(), 1)
        @test test_model(BosonBath(OhmicSpectralDensity(2.5, 0.1), 10), 10)
        @test test_model(SpinBoson(DebyeSpectralDensity(0.25, 0.5), 10, 1.0, 1.0), 10)
        @test test_model(OuyangModelOne(), 1)
        @test test_model(GatesHollowayElbow(), 2)
        @test test_model(MiaoSubotnik(Γ=0.1), 1)
        @test test_model(AnanthModelOne(), 1)
        @test test_model(AnanthModelTwo(), 1)
        @test test_model(ErpenbeckThoss(Γ=2.0), 1)
        @test test_model(WideBandBath(ErpenbeckThoss(Γ=2.0); step=0.1, bandmin=-1.0, bandmax=1.0), 1)
        @test test_model(WideBandBath(GatesHollowayElbow(); step=0.1, bandmin=-1.0, bandmax=1.0), 2)
        @test test_model(AndersonHolstein(ErpenbeckThoss(Γ=2.0), TrapezoidalRule(10, -1, 1)), 1)
        @test test_model(AndersonHolstein(ErpenbeckThoss(Γ=2.0), ShenviGaussLegendre(10, -1, 1)), 1)
        @test test_model(AndersonHolstein(GatesHollowayElbow(), ShenviGaussLegendre(10, -1, 1)), 2)
    end
end

if GROUP == "All" || GROUP == "Friction"
    @testset "FrictionModels" begin
        @test test_model(CompositeFrictionModel(Free(2), ConstantFriction(2, fill(1, 6,6))), 3)
        @test test_model(CompositeFrictionModel(Free(3), RandomFriction(3)), 3)
        # for sub in subtypes(NQCModels.FrictionModels.ElectronicFrictionProvider) # Every ElectronicFrictionProvider must have a friction_atoms field to select which parts of a system friction is applied to. 
        #     @test :friction_atoms ∈ fieldnames(sub) # This should go in FrictionProviders
        # end
    end
end

if GROUP == "All" || GROUP == "JuLIP"
    @testset "JuLIP" begin
        using JuLIP: JuLIP
        at = JuLIP.bulk(:Si, cubic=true)
        deleteat!(at, 1)
        JuLIP.set_calculator!(at, JuLIP.StillingerWeber())
        model = AdiabaticModels.JuLIPModel(at)
        @test test_model(model, length(at))
    end
end
