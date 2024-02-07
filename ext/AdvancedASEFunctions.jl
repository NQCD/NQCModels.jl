module AdvancedASEFunctions
using NQCModels, PyCall

"""
This module contains methods related to the NQCModels ASE interface that need access to Python types. (e.g. constraint checking) 
"""

function NQCModels.mobileatoms(model::NQCModels.AdiabaticModels.AdiabaticASEModel, n::Int)
	ase=pyimport("ase")
	constraints_FixAtoms=isa.(model.atoms.constraints, typeof(ase.constraints.FixAtoms))
	return symdiff(1:length(model.atoms), [constraint.get_indices() .+ 1 for constraint in model.atoms.constraints[constraints_FixAtoms]]...)
end

end