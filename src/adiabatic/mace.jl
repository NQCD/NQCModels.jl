using PyCall
using Unitful
using UnitfulAtomic
using NQCBase
using Statistics

mace_data = pyimport("mace.data")
torch = pyimport("torch")
mace_tools = pyimport("mace.tools")
numpy = pyimport("numpy")

mutable struct MACEPredictionCache{T}
    energies::Vector{Vector{T}}
    node_energy::Vector{Matrix{T}}
    forces::Vector{AbstractArray{T, 3}}
    stress::Vector{AbstractArray{T, 3}}
    input_structures::Vector
end

"""
MACE interface with support for ensemble of models and batch size selection for potentially faster inference. 


"""
struct MACEModel{A} <: AdiabaticModel
    model_paths::Vector{String}
    models::Vector
    device::Vector{String}
    default_dtype::A
    batch_size::Union{Int, Nothing}
    cutoff_radius::A
    last_eval_cache::MACEPredictionCache
    z_table::PyObject
end

# ToDo: Nice constructor for MACEModel and simpler input for single model. 

function MACEModel(model_paths::Vector{String}, device::Union{String, Vector{String}}="cpu", default_dtype::Union{Float64, Float32}=Float64, batch_size::Int=1)
    # Assign device to all models if only one device is given
    isa(device, String) ? device = [device for _ in 1:length(model_paths)] : nothing
    # Check selected device types are available
    for dev in device
        if split(dev, ":")[1] == "cuda" && torch.cuda.is_available()
            @debug "CUDA device available, using GPU."
            if length(split(dev, ":")) == 2
                torch.cuda.device_count() < parse(Int, split(dev, ":")[2]) || throw(ArgumentError("CUDA device index out of range."))
            end
        elseif dev == "mps" && torch.backends.mps.is_available()
            @debug "MPS device available, using GPU."
        else
            @warn "No CUDA device available, falling back to CPU."
            dev = "cpu"
        end
    end
    # Set default dtype for torch
    dtypes_julia_python = Dict(Float32 => torch.float32, Float64 => torch.float64)
    torch.set_default_dtype(dtypes_julia_python[default_dtype])
    
    # Load MACE models
    models = []
    for file_path in model_paths
        try
            model=torch.load(f=file_path, map_location=device)
            model=model.to(device)
            push!(models, model)
        catch
            @error "Error loading model from file: $file_path"
        end
    end

    # Store in models vector
    for model_path in model_paths
        push!(models, pyimport("torch").jit.load(model_path))
    end

    # Check cutoff radii are identical
    cutoff_radii = [model.r_max.cpu() for model in models]
    if length(unique(cutoff_radii)) > 1
        @warn "Cutoff radii are not identical for all models."
    end
    cutoff_radius = unique(cutoff_radii)[1]

    # Build z-table (needs PyVector representation)
    z_table = mace_tools.utils.AtomicNumberTable(PyVector([Int(Z) for Z in models[1].atomic_numbers]))

    # Freeze parameters
    for model in models
        for parameter in model.parameters()
            parameter.requires_grad = false
        end
    end

    # Initialise an evaluation cache
    starter_mace_cache = MACEPredictionCache(
        [zeros(default_dtype, length(models))], # Energies
        [zeros(default_dtype, (1, length(models)))], # Node energies
        [zeros(default_dtype, (1, 1, length(models)))], # Forces
        [zeros(default_dtype, (3, 3, length(models)))], # Stresses
        [], # Input structures
    )

    return MACEModel(model_paths, models, device, default_dtype, batch_size, cutoff_radius, starter_mace_cache, z_table)
end

# ToDo: Entry point into MACE's configuration data handling - py-Configuration from NQCD objects.
"""
    mace_configuration_from_nqcd_configuration(
    atoms::Atoms,  
    cell::Union{InfiniteCell, PeriodicCell},
    R::AbstractMatrix,
)

Converter into a single mace.data.utils.Configuration to make use of MACE's data loading. 
"""
function mace_configuration_from_nqcd_configuration(
    atoms::Atoms,  
    cell::Union{InfiniteCell, PeriodicCell},
    R::AbstractMatrix,
)
    if isa(cell, InfiniteCell)
        pbc = zeros(Bool, size(R, 1))
        cell_array = zeros(eltype(R), size(R, 1), size(R, 1))
    elseif isa(cell, PeriodicCell)
        pbc = cell.periodicity
        cell_array = Matrix{eltype(R)}(@. ustrip(auconvert(u"Å", cell.vectors')))
    end

    ase_positions = @. ustrip(auconvert(u"Å", R')) 

    config = mace_data.utils.Configuration(
        atomic_numbers = PyVector(atoms.numbers), # needs to be a list 
        positions = ase_positions, # Convert from atomic units to Ångström
        energy = zero(eltype(R)), # scalar
        forces = zeros(eltype(R), size(R')), # N_atoms * N_dofs
        stress = zeros(eltype(R), 2*size(R, 1)), # 2*N_dofs
        virials = zeros(eltype(R), (size(R, 1), size(R, 1))), # N_dofs * N_dofs
        dipole = zeros(eltype(R), size(R, 1)), # N_dofs
        charges = zeros(eltype(R), size(R, 2)), # N_dofs
        weight = one(eltype(R)), # scalar identity weight
        energy_weight = zero(eltype(R)),
        forces_weight = zero(eltype(R)),
        stress_weight = zero(eltype(R)),
        virials_weight = zero(eltype(R)),
        config_type = "Default",
        pbc = pbc,
        cell = cell_array,
    )
    return config
end

# ToDo: Evaluation function that handles model evaluation and caching of results. 

function predict!(
    mace_interface::MACEModel, 
    atoms::Vector{Atoms},
    R::Vector{AbstractMatrix},
    cell::Vector{Union{InfiniteCell, PeriodicCell}},
    )
    if R != mace_interface.last_eval_cache.input_structures
        # Create MACE atomicdata representation
        dataset = Vector{PyObject}(undef, length(R))
        @views for i in axes(R, 3)
            config = mace_configuration_from_nqcd_configuration(atoms[i], cell[i], R[i])
            dataset[i] = mace_data.utils.AtomicData.from_config(config, mace_interface.z_table, mace_interface.cutoff_radius)
        end
        # Initialise DataLoader
        batch_size = mace_interface.batch_size === nothing ? length(dataset) : mace_interface.batch_size # Ensure there is a batch size
        mace_DataLoader = mace_tools.torch_geometric.dataloader.DataLoader(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = false,
            drop_last=false,
        )
        # Update evaluation cache for outputs
        mace_interface.last_eval_cache.input_structures = R 
        mace_interface.last_eval_cache.energies = [zeros(mace_interface.default_dtype, length(mace_interface.models)) for _ in 1:length(dataset)]
        mace_interface.last_eval_cache.node_energy = [zeros(mace_interface.default_dtype, (size(i, 2), length(mace_interface.models))) for i in dataset]
        mace_interface.last_eval_cache.forces = [zeros(mace_interface.default_dtype, (size(i, 2), size(i, 1), length(mace_interface.models))) for i in dataset] 
        mace_interface.last_eval_cache.stress = [zeros(mace_interface.default_dtype, (3, 3, length(mace_interface.models))) for _ in 1:length(dataset)]

        # Iterate through dataloader and evaluate each model
        for batch in mace_DataLoader
            for (index, model) in enumerate(mace_interface.models)
                # Place copy of batch on model device
                clone = batch.clone().to(mace_interface.device[index])
                # Evaluate model
                model_output = model(clone.to_dict(), compute_stress=true)
                # Update evaluation cache
                setindex!.(mace_interface.last_eval_cache.energy, model_output["energy"].cpu().detach().numpy(), index)
                # Split forces according to batching 
                #! Check how well this performs and whether this actually saves memory
                energies = model_output["energies"].cpu().detach().numpy() 
                forces = model_output["forces"].cpu().detach().numpy() 
                splitting = clone.ptr.cpu().detach().numpy() # Array of batch item bounds in output arrays
                for batch_structure in 2:length(splitting)
                    mace_interface.last_eval_cache.energies[batch_structure-1] = energies[batch_structure-1]
                    @views mace_interface.last_eval_cache.forces[batch_structure][:, :, index] .= forces[splitting[batch_structure-1]:splitting[batch_structure], :]
                end
            end
        end
    end
end

# Same atoms and cell for multiple structures
predict!(mace_interface::MACEModel, atoms::Atoms, R::Vector{AbstractMatrix}, cell::Union{InfiniteCell, PeriodicCell}) = predict!(mace_interface, [atoms for _ in 1:length(R)], R, [cell for _ in 1:length(R)])
# Single structure version
predict!(mace_interface::MACEModel, atoms::Atoms, R::AbstractMatrix, cell::Union{InfiniteCell, PeriodicCell}) = predict!(mace_interface, [atoms], [R], [cell])




# ToDo: Methods using MACEPredictionCache that convert MACE outputs to NQCD's atomic unit scheme. Snip length 1 caches to the basic outputs instead of unnecessary vector wrapping. 
function get_energy_mean(mace_cache::MACEPredictionCache)
    mean_energies = zeros(eltype(mace_cache.energies[1]), length(mace_cache.energies))
    for index in eachindex(mace_cache.energies)
        mean_energies[index] = mean(austrip(mace_cache.energies[index] .* u"eV")) # Energy is given in eV
    end
    if length(mean_energies) == 1 
        return mean_energies[1]
    else
        return mean_energies # Return in Hartree
    end
end

function get_energy_variance(mace_cache::MACEPredictionCache)
    mean_energies = zeros(eltype(mace_cache.energies[1]), length(mace_cache.energies))
    for index in eachindex(mace_cache.energies)
        mean_energies[index] = var(austrip(mace_cache.energies[index] .* u"eV")) # Energy is given in eV
    end
    if length(mean_energies) == 1 
        return mean_energies[1]
    else
        return mean_energies # Return in Hartree^2
    end
end

function get_energy_ensemble(mace_cache::MACEPredictionCache)
    ensemble_energies = zeros(type(mace_cache.energies[1]), length(mace_cache.energies))
    for index in eachindex(mace_cache.energies)
        ensemble_energies[index] = austrip(mace_cache.energies[index] .* u"eV") # Energy is given in eV
    end
    if length(ensemble_energies) == 1 
        return ensemble_energies[1]
    else
        return ensemble_energies # Return in Hartree
    end
end

function get_forces_mean(mace_cache::MACEPredictionCache)
    mean_forces = zeros(type(mace_cache.forces[1]), size(mace_cache.forces[1]))
    for index in eachindex(mace_cache.forces)
        mean_forces[index] = dropdims(mean(austrip.(mace_cache.forces[index] .* u"eV/Å"); dims=3); dims=3)' # Force is given in eV/Å
    end
    if length(mean_forces) == 1 
        return mean_forces[1]
    else
        return mean_forces # Return in Hartree / Bohr
    end
end

function get_forces_variance(mace_cache::MACEPredictionCache)
    mean_forces = zeros(type(mace_cache.forces[1]), size(mace_cache.forces[1]))
    for index in eachindex(mace_cache.forces)
        mean_forces[index] = dropdims(var(austrip.(mace_cache.forces[index] .* u"eV/Å"); dims=3); dims=3)' # Force is given in eV/Å
    end
    if length(mean_forces) == 1 
        return mean_forces[1]
    else
        return mean_forces # Return in Hartree / Bohr^2
    end
end

function get_forces_ensemble(mace_cache::MACEPredictionCache)
    ensemble_forces = zeros(type(mace_cache.forces[1]), size(mace_cache.forces[1]))
    for index in eachindex(mace_cache.forces)
        ensemble_forces[index] = austrip.(mace_cache.forces[index] .* u"eV/Å") # Force is given in eV/Å
    end
    if length(ensemble_forces) == 1 
        return ensemble_forces[1]
    else
        return ensemble_forces # Return in Hartree / Bohr
    end
end

# ToDo: Potential and derivative for a single structure
function NQCModels.potential(model::MACEModel, atoms::Atoms, R::AbstractMatrix, cell::Union{InfiniteCell, PeriodicCell})
    # Evaluate model
    predict!(model, atoms, cat(R; dims=3), cell)
    # Return potential (mean is trivial here)
    return get_energy_mean(model.last_eval_cache)
end

function NQCModels.derivative(model::MACEModel, atoms::Atoms, R::AbstractMatrix, cell::Union{InfiniteCell, PeriodicCell})
    # Evaluate model
    predict!(model, atoms, cat(R; dims=3), cell)
    # Return derivative (mean is trivial)
    return -get_forces_mean(model.last_eval_cache)
end

function NQCModels.derivative!(model::MACEModel, D::AbstractMatrix, atoms::Atoms, R::AbstractMatrix, cell::Union{InfiniteCell, PeriodicCell})
    # Evaluate model
    predict!(model, atoms, cat(R; dims=3), cell)
    # Return derivative
    D .-= get_forces_mean(model.last_eval_cache)
end

# ToDo: Potential and derivative for multiple structures


# ToDo: Evaluation functions for the model which check whether the prediction is up to date and only evaluate if necessary.

#? Unsure whether to implement functions such as evaluate_forces(model, R) since predict!() isn't too difficult to understand. 

export predict!, get_energy_mean, get_energy_variance, get_energy_ensemble, get_forces_mean, get_forces_variance, get_forces_ensemble, MACEModel, MACEPredictionCache