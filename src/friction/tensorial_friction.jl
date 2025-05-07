


function friction(model::ConstantFriction{AbstractMatrix}, ::AbstractMatrix)
	return model.γ
end

function friction!(model::ConstantFriction{AbstractMatrix}, F::AbstractMatrix, ::AbstractMatrix)
	F .= model.γ
end

"""
	friction_matrix_indices(model, indices)

Returns the indices of the friction matrix corresponding to the given Atom indices.
"""
function friction_matrix_indices(indices, dofs)
	dof_range = collect(1:dofs)
	return vcat(broadcast(x -> x .+ dof_range, dofs .* (indices .- 1))...)
end

function friction!(model::TensorialFriction, F::AbstractMatrix, R::AbstractMatrix)
	indices = friction_matrix_indices(model.friction_atoms, NQCModels.ndofs(model))
	F[indices, indices] .= get_friction_matrix(model, R)
end


