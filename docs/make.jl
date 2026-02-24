using Documenter
using NQCModels
using NQCBase

makedocs(;
    sitename="NQCModels.jl",
    modules=[NQCModels],
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true",
    ),
    pages=[
        "Full dimensional models" => "fullsizemodels.md",
        "AtomsCalculators interoperability" => "atomscalculators.md",
    ],
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo="github.com/NQCD/NQCModels.jl",
    )
end
