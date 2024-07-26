module PythonCallASEext

using NQCModels: AdiabaticASEModel
using PythonCall
using Unitful: @u_str, ustrip
using UnitfulAtomic: austrip, auconvert


function NQCModels.potential(model::AdiabaticASEModel, R::AbstractMatrix)
    set_coordinates!(model, R)
    V = model.atoms.get_potential_energy()
    return austrip(pyconvert(eltype(R),V) * u"eV")
end

function NQCModels.derivative!(model::AdiabaticASEModel, D::AbstractMatrix, R::AbstractMatrix)
    set_coordinates!(model, R)
    D .= -pyconvert(Matrix{eltype(D)}, model.atoms.get_forces())'
    @. D = austrip(D * u"eV/Å")
    return D
end


"""
This module contains methods related to the NQCModels ASE interface that need access to Python types. (e.g. constraint checking)
"""

function NQCModels.mobileatoms(model::AdiabaticASEModel, n::Int)
	return symdiff(1:length(model.atoms), [pyconvert(Vector,constraint.get_indices()) .+ 1 for constraint in model.atoms.constraints]...)
end


end