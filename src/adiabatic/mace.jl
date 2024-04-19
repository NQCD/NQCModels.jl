__precompile__()
module MACE
"""
This module contains an interface to the MACE machine learning interatomic potential, borrowing heavily from eval_configs.py and the MACE ase calculator. 

MIT License

Copyright (c) 2022 ACEsuit/mace

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
"""

using PyCall
using Unitful
using UnitfulAtomic
using NQCBase
using Statistics
using ..NQCModels: NQCModels, AdiabaticModels.AdiabaticChemicalEnvironmentMLIP

const torch = PyNULL()
const mace_data = PyNULL()
const mace_tools = PyNULL()
const numpy = PyNULL()

function __init__()
    copy!(torch, pyimport("torch"))
    copy!(mace_data, pyimport("mace.data"))
    copy!(mace_tools, pyimport("mace.tools"))
    copy!(numpy, pyimport("numpy"))
end

mutable struct MACEPredictionCache{T}
    energies::Vector{Vector{T}}
    node_energy::Vector{Matrix{T}}
    forces::Vector{<:AbstractArray{T, 3}}
    stress::Vector{<:AbstractArray{T, 3}}
    input_structures::Vector
end

"""
MACE interface with support for ensemble of models and batch size selection for potentially faster inference. 

DOFs currently hardcoded at 3. 

"""
struct MACEModel{T} <: AdiabaticChemicalEnvironmentMLIP
    model_paths::Vector{String}
    models::Vector
    device::Vector{String}
    default_dtype::Type{T}
    batch_size::Union{Int, Nothing}
    cutoff_radius::T
    last_eval_cache::MACEPredictionCache
    z_table
    atoms::Atoms
    cell::PeriodicCell
    ndofs::Int
end

NQCModels.ndofs(model::MACEModel) = 3
NQCModels.dofs(model::MACEModel) = Base.OneTo(3)

# ToDo: Nice constructor for MACEModel and simpler input for single model. 

"""
    MACEModel(
    model_paths::Vector{String}, 
    device::Union{String, Vector{String}}="cpu", 
    default_dtype::Type=Float64, 
    batch_size::Int=1,
    atoms,
    cell,
)

Interface to MACE machine learning interatomic potentials with support for ensemble of models and batch size selection for potentially faster inference.

# Arguments
- `model_paths::Vector{String}`: Model paths to load. If multiple paths are given, an ensemble of models is used and mean energies/forces used to propagate dynamics. 
- `device::Union{String, Vector{String}}`: Device to use for inference. If a single device is given, all models will be loaded on that device. If multiple devices are given, each model will be loaded on the corresponding device. Default is `"cpu"`.
- `default_dtype::Type`: Default data type for PyTorch (usually Float32)
- `batch_size::Int`: Batch size for inference. Can be set to `nothing` to always adapt batch size to the number of structures provided. Ring-polymer methods benefit from this. 
- `atoms`: Atoms object for the system. `predict!` can be called with a Vector of different Atoms objects to evaluate different structures. 
- `cell`: Cell object for the system. `predict!` can be called with a Vector of different Cell objects to evaluate different structures.
"""
function MACEModel(
    atoms::Atoms,
    cell::AbstractCell,
    model_paths::Vector{String}, 
    device::Union{String, Vector{String}}="cpu", 
    default_dtype::Type=Float32, 
    batch_size::Int=1,
)
    # Assign device to all models if only one device is given
    isa(device, String) ? device = [device for _ in 1:length(model_paths)] : nothing
    # Check selected device types are available
    for dev in device
        if split(dev, ":")[1] == "cuda"
            if torch.cuda.is_available()
                @debug "CUDA device available, using GPU."
            else
                @warn "CUDA device not available, falling back to CPU."
                dev = "cpu"
            end
            if length(split(dev, ":")) == 2
                torch.cuda.device_count() < parse(Int, split(dev, ":")[2]) || throw(ArgumentError("CUDA device index out of range."))
            end
        elseif dev == "mps"
            if torch.backends.mps.is_available()
                @debug "MPS device available, using GPU."
            else
                @warn "MPS device not available, falling back to CPU."
                dev = "cpu"
            end
        else
            @debug "CPU selected as torch.device"
            dev = "cpu"
        end
    end
    # Set default dtype for torch
    dtypes_julia_python = Dict{Type, Any}(Float32 => torch.float32, Float64 => torch.float64)
    torch.set_default_dtype(dtypes_julia_python[default_dtype])
    
    # Load MACE models
    models = []
    for (i,file_path) in enumerate(model_paths)
        try
            model=torch.load(f=file_path, map_location=device[i])
            model=model.to(device[i])
            push!(models, model)
        catch e
            throw(e)
            @error "Error loading model from file: $file_path"
        end
    end

    # Check cutoff radii are identical
    cutoff_radii = [model.r_max.cpu().item() for model in models]
    if length(unique(cutoff_radii)) > 1
        @warn "Cutoff radii are not identical for all models."
    end
    cutoff_radius = convert(default_dtype, unique(cutoff_radii)[1])

    # Build z-table (needs PyVector representation)
    z_table = mace_tools.utils.AtomicNumberTable(PyVector(py"[int(Z) for Z in $models[0].atomic_numbers]"))

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

    return MACEModel(model_paths, models, device, default_dtype, batch_size, cutoff_radius, starter_mace_cache, z_table, atoms, cell, 3)
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
    atoms::Union{Vector{<:Atoms}, Atoms},
    R::Vector{<:AbstractMatrix},
    cell::Union{Vector{<:AbstractCell}, AbstractCell},
    )
    if R != mace_interface.last_eval_cache.input_structures
        # Create MACE atomicdata representation
        dataset = Vector{Any}(undef, length(R))
        isa(cell, AbstractCell) ? cell = [cell for _ in 1:length(R)] : nothing
        isa(atoms, Atoms) ? atoms = [atoms for _ in 1:length(R)] : nothing
        for i in eachindex(R)
            config = mace_configuration_from_nqcd_configuration(atoms[i], cell[i], R[i])
            dataset[i] = mace_data.AtomicData.from_config(config, mace_interface.z_table, mace_interface.cutoff_radius)
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
        mace_interface.last_eval_cache.energies = [zeros(mace_interface.default_dtype, length(mace_interface.models)) for _ in R]
        mace_interface.last_eval_cache.node_energy = [zeros(mace_interface.default_dtype, (size(i, 2), length(mace_interface.models))) for i in R]
        mace_interface.last_eval_cache.forces = [zeros(mace_interface.default_dtype, (size(i, 2), size(i, 1), length(mace_interface.models))) for i in R] 
        mace_interface.last_eval_cache.stress = [zeros(mace_interface.default_dtype, (3, 3, length(mace_interface.models))) for _ in R]

        # Iterate through dataloader and evaluate each model
        for (batch_index, batch) in enumerate(mace_DataLoader)
            evalcache_index = (batch_index - 1) * batch_size + 1 # Pointer to the start of the batch in the output arrays
            for (model_index, model) in enumerate(mace_interface.models)
                # Place copy of batch on model device
                clone = batch.clone().to(mace_interface.device[model_index])
                # Evaluate model
                model_output = model(clone.to_dict(), compute_stress=true)
                # Split according to batching
                #! Check how well this performs and whether this actually saves memory
                energies = model_output["energy"].cpu().detach().numpy() 
                forces = model_output["forces"].cpu().detach().numpy() 
                splitting = clone.ptr.cpu().detach().numpy() .+1 # Array of batch item bounds in output arrays, +1 due to Julia-Python conversion
                for structure_index in 2:length(splitting)
                    mace_interface.last_eval_cache.energies[evalcache_index+structure_index-1][model_index] = energies[structure_index-1]
                    @views mace_interface.last_eval_cache.forces[evalcache_index+structure_index-1][:, :, model_index] .= forces[splitting[structure_index-1]:splitting[structure_index]-1, :] # last index -1 because Julia includes last index in a slice
                end
            end
        end
    end
end

function predict(
    mace_interface::MACEModel, 
    atoms::Union{Vector{<:Atoms}, Atoms},
    R::Vector{<:AbstractMatrix},
    cell::Union{Vector{<:AbstractCell}, Atoms},
    )
    predict!(mace_interface, atoms, R, cell)
    return deepcopy(mace_interface.last_eval_cache)
end

# ToDo: Methods using MACEPredictionCache that convert MACE outputs to NQCD's atomic unit scheme. Snip length 1 caches to the basic outputs instead of unnecessary vector wrapping. 
function get_energy_mean(mace_cache::MACEPredictionCache)
    mean_energies = zeros(eltype(mace_cache.energies[1]), length(mace_cache.energies))
    for index in eachindex(mace_cache.energies)
        mean_energies[index] = mean(austrip.(mace_cache.energies[index] .* u"eV")) # Energy is given in eV
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
        mean_energies[index] = var(austrip.(mace_cache.energies[index] .* u"eV")) # Energy is given in eV
    end
    if length(mean_energies) == 1 
        return mean_energies[1]
    else
        return mean_energies # Return in Hartree^2
    end
end

function get_energy_ensemble(mace_cache::MACEPredictionCache)
    ensemble_energies = Vector{typeof(mace_cache.energies[1])}(undef , length(mace_cache.energies))
    for index in eachindex(mace_cache.energies)
        ensemble_energies[index] = austrip.(mace_cache.energies[index] .* u"eV") # Energy is given in eV
    end
    if length(ensemble_energies) == 1 
        return ensemble_energies[1]
    else
        return ensemble_energies # Return in Hartree
    end
end

function get_forces_mean(mace_cache::MACEPredictionCache)
    mean_forces = Vector{Matrix{eltype(mace_cache.forces[1])}}(undef,  length(mace_cache.forces))
    for index in eachindex(mace_cache.forces)
        mean_forces[index] = permutedims(dropdims(mean(austrip.(mace_cache.forces[index] .* u"eV/Å"); dims=3); dims=3), (2,1)) # Force is given in eV/Å
    end
    if length(mean_forces) == 1 
        return mean_forces[1]
    else
        return mean_forces # Return in Hartree / Bohr
    end
end

function get_forces_variance(mace_cache::MACEPredictionCache)
    mean_forces = Vector{Matrix{eltype(mace_cache.forces[1])}}(undef,  length(mace_cache.forces))
    for index in eachindex(mace_cache.forces)
        mean_forces[index] = permutedims(dropdims(var(austrip.(mace_cache.forces[index] .* u"eV/Å"); dims=3); dims=3), (2,1)) # Force is given in eV/Å
    end
    if length(mean_forces) == 1 
        return mean_forces[1]
    else
        return mean_forces # Return in Hartree / Bohr^2
    end
end

function get_forces_ensemble(mace_cache::MACEPredictionCache)
    ensemble_forces = Vector{typeof(mace_cache.forces[1])}(undef, length(mace_cache.forces))
    for index in eachindex(mace_cache.forces)
        ensemble_forces[index] =  zeros(eltype(mace_cache.forces[index]), size(mace_cache.forces[index])[[2,1,3]])
        permutedims!(ensemble_forces[index], austrip.(mace_cache.forces[index] .* u"eV/Å"), (2,1,3)) # Force is given in eV/Å
    end
    if length(ensemble_forces) == 1 
        return ensemble_forces[1]
    else
        return ensemble_forces # Return in Hartree / Bohr
    end
end

# ToDo: Potential and derivative for a single structure
# Passthrough for NQCDynamics Calculators
NQCModels.potential(model::MACEModel, R::AbstractMatrix) = NQCModels.potential(model, model.atoms, R, model.cell)
NQCModels.derivative(model::MACEModel, R::AbstractMatrix) = NQCModels.derivative(model, model.atoms, R, model.cell)
NQCModels.derivative!(model::MACEModel, D::AbstractMatrix, R::AbstractMatrix) = NQCModels.derivative!(model, D, model.atoms, R, model.cell)

# Single structure evaluation with custom atoms and cell. 
function NQCModels.potential(model::MACEModel, atoms::Atoms, R::AbstractMatrix, cell::AbstractCell)
    # Evaluate model
    predict!(model, atoms, [R], cell)
    # Return potential (mean is trivial here)
    return get_energy_mean(model.last_eval_cache)
end

function NQCModels.potential(model::MACEModel, atoms::Atoms, R::Vector{<:AbstractMatrix}, cell::AbstractCell)
    # Evaluate model
    predict!(model, atoms, R, cell)
    # Return potential (mean is trivial here)
    return get_energy_mean(model.last_eval_cache)
end

function NQCModels.derivative(model::MACEModel, atoms::Atoms, R::AbstractMatrix, cell::Union{InfiniteCell, PeriodicCell})
    # Evaluate model
    predict!(model, atoms, [R], cell)
    # Return derivative (mean is trivial)
    return -get_forces_mean(model.last_eval_cache)
end

function NQCModels.derivative(model::MACEModel, atoms::Atoms, R::Vector{<:AbstractMatrix}, cell::Union{InfiniteCell, PeriodicCell})
    # Evaluate model
    predict!(model, atoms, R, cell)
    # Return derivative (mean is trivial)
    return .- get_forces_mean(model.last_eval_cache)
end

function NQCModels.derivative!(model::MACEModel, D::AbstractMatrix, atoms::Atoms, R::AbstractMatrix, cell::Union{InfiniteCell, PeriodicCell})
    # Evaluate model
    predict!(model, atoms, [R], cell)
    # Return derivative
    D .-= get_forces_mean(model.last_eval_cache)
end

# ToDo: Potential and derivative for multiple structures


# ToDo: Evaluation functions for the model which check whether the prediction is up to date and only evaluate if necessary.

#? Unsure whether to implement functions such as evaluate_forces(model, R) since predict!() isn't too difficult to understand. 

export predict, predict!, get_energy_mean, get_energy_variance, get_energy_ensemble, get_forces_mean, get_forces_variance, get_forces_ensemble, MACEModel, MACEPredictionCache
end
