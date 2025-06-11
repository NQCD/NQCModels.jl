using Test
using NQCModels

using Unitful
using LinearAlgebra: Hermitian, diagind

@testset "TrapezoidalRule" begin
    bath = NQCModels.TrapezoidalRule(50, -10, 10)
    n = NQCModels.nstates(bath)
    out = Hermitian(zeros(n+1, n+1))

    couplings_rescale = 1.0

    NQCModels.BathDiscretisations.fillbathstates!(out, bath)
    @test all(out[diagind(out)[2:end]] .== bath.bathstates)
    NQCModels.BathDiscretisations.fillbathcoupling!(out, 1.0, bath, couplings_rescale)
    @test all(isapprox.(out[1,2:end], 1.0 / sqrt(50/20)))
    @test all(isapprox.(out[2:end,1], 1.0 / sqrt(50/20)))
end

@testset "ShenviGaussLegendre" begin
    bath = NQCModels.BathDiscretisations.ShenviGaussLegendre(20, -10, 10.0)

    n = NQCModels.nstates(bath)
    out = Hermitian(zeros(n+1, n+1))

    couplings_rescale = 1.0

    NQCModels.BathDiscretisations.fillbathstates!(out, bath)
    NQCModels.BathDiscretisations.fillbathcoupling!(out, 1.0, bath, couplings_rescale)
end

@testset "ReferenceGaussLegendre" begin
    bath = NQCModels.BathDiscretisations.ReferenceGaussLegendre(20, -10, 10.0)

    n = NQCModels.nstates(bath)
    out = Hermitian(zeros(n+1, n+1))

    couplings_rescale = 1.0

    NQCModels.BathDiscretisations.fillbathstates!(out, bath)
    NQCModels.BathDiscretisations.fillbathcoupling!(out, 1.0, bath, couplings_rescale)
end

@testset "FullGaussLegendre" begin
    bath = NQCModels.BathDiscretisations.FullGaussLegendre(20, -10, 10.0)

    n = NQCModels.nstates(bath)
    out = Hermitian(zeros(n+1, n+1))

    couplings_rescale = 1.0

    NQCModels.BathDiscretisations.fillbathstates!(out, bath)
    NQCModels.BathDiscretisations.fillbathcoupling!(out, 1.0, bath, couplings_rescale)
end

@testset "GapGaussLegendre" begin
    bath = NQCModels.BathDiscretisations.GapGaussLegendre(20, -10, 10.0, 2.0)

    n = NQCModels.nstates(bath)
    out = Hermitian(zeros(n+1, n+1))
    couplings_rescale = 1.0

    NQCModels.BathDiscretisations.fillbathstates!(out, bath)
    NQCModels.BathDiscretisations.fillbathcoupling!(out, 1.0, bath, couplings_rescale)
end


@testset "GapTrapezoidalRule" begin
    bath = NQCModels.BathDiscretisations.GapTrapezoidalRule(20, -10, 10.0, 2.0)

    n = NQCModels.nstates(bath)
    out = Hermitian(zeros(n+1, n+1))
    couplings_rescale = 1.0

    NQCModels.BathDiscretisations.fillbathstates!(out, bath)
    NQCModels.BathDiscretisations.fillbathcoupling!(out, 1.0, bath, couplings_rescale)
end