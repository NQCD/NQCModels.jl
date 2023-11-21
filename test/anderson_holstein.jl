"""
This test script originally written by James Gardner, University of Warwick, UK

Comments and changes from Xuexun Lu on Tue, 21 Nov 2023 in University of Warwick, UK


If you wanted to test your local branch NQCModels.jl, you have to set the path of NQCModels.jl locally

    i.e. In Julia pkg> mode, type 
                                dev "/User/uxxxxxx/Desktop/NQCModels.jl"
                                
    Just attach the local package to Julia. Otherwise, just use the default(downloaded) NQCModels.jl in Julia pkg> mode.

    if you want to switch back to the default NQCModels.jl, type in Julia pkg> mode
                                pkg> free NQCModels
"""


using Test
using NQCModels

m = ErpenbeckThoss(;Γ=0.1)
b = TrapezoidalRule(10, -10, 10)
@testset "nelectrons" begin
    model = AndersonHolstein(m, b) # fermilevel defaults to zero
    @test NQCModels.nelectrons(model) == 5

    for (n, f) in zip([1, 6, 10], [-10, 2, 10])
        model = AndersonHolstein(m, b; fermi_level=f)
        @test NQCModels.nelectrons(model) == n
    end
end
