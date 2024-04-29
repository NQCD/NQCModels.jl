module AdvancedASEFunctions
using NQCModels, PyCall

const ase = PyNULL()

function __init__()
    copy!(ase, pyimport("ase"))
end

"""
This module contains methods related to the NQCModels ASE interface that need access to Python types. (e.g. constraint checking)
"""

function NQCModels.mobileatoms(model::NQCModels.AdiabaticModels.AdiabaticASEModel, n::Int)
	constraints_FixAtoms=isa.(model.atoms.constraints, typeof(ase.constraints.FixAtoms))
	ase_constraints = [constraint for constraint in model.atoms.constraints[constraints_FixAtoms]]
	if length(ase_constraints) == 0
        return 1:length(model.atoms)
	else
        return symdiff(1:length(model.atoms), [constraint.get_indices() .+ 1 for constraint in constraints]...)
    end
end

end
