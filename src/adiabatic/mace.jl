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

using PythonCall
using DLPack
using Unitful
using UnitfulAtomic
using NQCBase
using Statistics
using ..NQCModels: NQCModels, AdiabaticModels.AdiabaticChemicalEnvironmentMLIP

const torch = Ref{Py}()
const mace_data = Ref{Py}()
const mace_tools = Ref{Py}()
const numpy = Ref{Py}()

function __init__()
    torch[] = pyimport("torch")
    mace_data[] = pyimport("mace.data")
    mace_tools[] = pyimport("mace.tools")
    numpy[] = pyimport("numpy")
end

"""
Cache which stores the results of the last evaluation of the MACE model.
This is used to speed up concurrent energy and force evaluations, as well as to give access to ensemble predictions.

Use e.g. `get_energy_mean(MACEModel.last_eval_cache)` to access results of the last prediction made by the model.
"""
mutable struct MACEPredictionCache{T}
    energies::Vector{Vector{T}}
    node_energy::Vector{Matrix{T}}
    forces::Vector{<:AbstractArray{T,3}}
    stress::Vector{<:AbstractArray{T,3}}
    input_structures::Vector
end

"""
MACE interface with support for ensemble of models and batch size selection for potentially faster inference.

DOFs currently hardcoded at 3.

# Fields
- `model_paths::Vector{String}`: Model paths to load. If multiple paths are given, an ensemble of models is used and mean energies/forces used to propagate dynamics.
- `models::Vector`: Loaded models.
- `device::Vector{String}`: Device to use for inference. If a single device is given, all models will be loaded on that device. If multiple devices are given, each model will be loaded on the corresponding device. Default is `"cpu"`.
- `default_dtype::Type{T}`: Default data type for inference. Default is `Float32`.
- `batch_size::Union{Int,Nothing}`: Batch size for inference. Default is `1`.
- `cutoff_radius::T`: Cutoff radius for the models.
- `last_eval_cache::MACEPredictionCache`: Cache for the last evaluation.
- `z_table`: Table of atomic numbers.
- `atoms::Atoms`: Atoms object for the system.
- `cell::PeriodicCell`: Periodic cell for the system.
- `ndofs::Int`: Number of degrees of freedom in the system.
- `mobile_atoms::Vector{Int}`: Indices of mobile atoms in the system.
"""
struct MACEModel{T} <: AdiabaticChemicalEnvironmentMLIP
    model_paths::Vector{String}
    models::Vector
    device::Vector{String}
    default_dtype::Type{T}
    batch_size::Union{Int,Nothing}
    cutoff_radius::T
    last_eval_cache::MACEPredictionCache
    z_table
    atoms::Atoms
    cell::AbstractCell
    ndofs::Int
    mobile_atoms::Vector{Int}
end

NQCModels.ndofs(model::MACEModel) = 3
NQCModels.dofs(model::MACEModel) = Base.OneTo(3)
mobileatoms(model::MACEModel, n::Int) = model.mobile_atoms # Return all mobile atoms in the system.

# ToDo: Nice constructor for MACEModel and simpler input for single model.

"""
    MACEModel(
    atoms::Atoms,
    cell::AbstractCell,
    model_paths::Vector{String};
    device::Union{String,Vector{String}}="cpu",
    default_dtype::Type=Float32,
    batch_size::Int=1,
    mobile_atoms::Vector{Int}=collect(1:length(atoms)),
)
    model_paths::Vector{String},
    device::Union{String, Vector{String}}="cpu",
    default_dtype::Type=Float64,
    batch_size::Int=1,
    atoms,
    cell,
)

Interface to MACE machine learning interatomic potentials with support for ensemble of models and batch size selection for potentially faster inference.

# Arguments
- `atoms`: Atoms object for the system. `predict!` can be called with a Vector of different Atoms objects to evaluate different structures.
- `cell`: Cell object for the system. `predict!` can be called with a Vector of different Cell objects to evaluate different structures.
- `model_paths::Vector{String}`: Model paths to load. If multiple paths are given, an ensemble of models is used and mean energies/forces used to propagate dynamics.
- `device::Union{String, Vector{String}}`: Device to use for inference. If a single device is given, all models will be loaded on that device. If multiple devices are given, each model will be loaded on the corresponding device. Default is `"cpu"`.
- `default_dtype::Type`: Default data type for PyTorch (usually Float32)
- `batch_size::Int`: Batch size for inference. Can be set to `nothing` to always adapt batch size to the number of structures provided. Ring-polymer methods benefit from this.
- `mobile_atoms::Vector{Int}`: Indices of mobile atoms in the system. Default is all atoms.
"""
function MACEModel(
    atoms::Atoms,
    cell::AbstractCell,
    model_paths::Vector{String};
    device::Union{String,Vector{String}}="cpu",
    default_dtype::Type=Float32,
    batch_size::Int=1,
    mobile_atoms::Vector{Int}=collect(1:length(atoms)),
)
    # Assign device to all models if only one device is given
    isa(device, String) ? device = [device for _ in 1:length(model_paths)] : nothing
    # Check selected device types are available
    for dev in device
        if split(dev, ":")[1] == "cuda"
            if pyconvert(Bool, torch[].backends.cuda.is_built())
                @debug "CUDA device available, using GPU."
            else
                @warn "CUDA device not available, falling back to CPU."
                dev = "cpu"
            end
            if length(split(dev, ":")) == 2
                pyconvert(Int, torch[].cuda.device_count()) < parse(Int, split(dev, ":")[2]) || throw(ArgumentError("CUDA device index out of range."))
            end
        elseif dev == "mps"
            if pyconvert(Bool, torch[].backends.mps.is_built())
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
    dtypes_julia_python = Dict{Type,Any}(Float32 => torch[].float32, Float64 => torch[].float64)
    torch[].set_default_dtype(dtypes_julia_python[default_dtype])

    # Load MACE models
    models = []
    for (i, file_path) in enumerate(model_paths)
        try
            model = torch[].load(f=file_path, map_location=device[i])
            model = model.to(device[i])
            # Check if any rogue atoms contained in base structure
            any(sort(unique(atoms.numbers)) ∉ Vector(from_dlpack(model.atomic_numbers))) || throw(ArgumentError("Example structure contains atom types not present in MACE model $(i)."))
            # Model is safe to add to ensemble
            push!(models, model)
        catch e
            throw(e)
            @error "Error loading model from file: $file_path"
        end
    end

    # Check cutoff radii are identical
    cutoff_radii = [copy(Vector(from_dlpack(model.r_max))[]) for model in models]
    if length(unique(cutoff_radii)) > 1
        @warn "Cutoff radii are not identical for all models."
    end
    cutoff_radius = convert(default_dtype, unique(cutoff_radii)[1])

    # Build z-table
    z_table = mace_tools[].utils.AtomicNumberTable(sort(unique(atoms.numbers)))

    # Freeze parameters
    for model in models
        for parameter in model.parameters()
            parameter.requires_grad = false
        end
    end

    # Initialise an evaluation cache
    starter_mace_cache = MACEPredictionCache(
        [[convert(default_dtype,1.0)]], # Energies
        [hcat(convert(default_dtype,1.0))], # Node energies
        [[convert(default_dtype,1.0);;;]], # Forces
        [[convert(default_dtype,1.0);;;]], # Stresses
        [], # Input structures
    )

    return MACEModel(model_paths, models, device, default_dtype, batch_size, cutoff_radius, starter_mace_cache, z_table, atoms, cell, 3, mobile_atoms)
end

function Base.show(io::IO, model::MACEModel)
    print(
        io,
        "MACEModel{$(model.default_dtype)} with $(length(model.models)) models:\n",
        "\tDevice(s): $(model.device) \n",
        "\tBatch size: $(model.batch_size) \n"
    )
end

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
    cell::AbstractCell,
    R::AbstractMatrix;
    dtype::Type=Float64,
)
    if eltype(R) != dtype
        R = convert(Matrix{dtype}, R)
    end
    if isa(cell, InfiniteCell)
        pbc = zeros(Bool, size(R, 1))
        cell_array = zeros(dtype, size(R, 1), size(R, 1))
    elseif isa(cell, PeriodicCell)
        pbc = cell.periodicity
        cell_array = Matrix{eltype(R)}(@. ustrip(auconvert(u"Å", cell.vectors')))
    end

    ase_positions = @. ustrip(auconvert(u"Å", R'))

    config = mace_data[].utils.Configuration(
        atomic_numbers=PyList(atoms.numbers), # needs to be a list
        positions=numpy[].array(ase_positions), # Convert from atomic units to Ångström
        energy=Py(zero(eltype(R))), # scalar
        forces=numpy[].array(zeros(eltype(R), size(R'))), # N_atoms * N_dofs
        stress=pybuiltins.None, # Avoid unnecessary tensor overhead for functions not implemented in NQCD
        virials=pybuiltins.None, # Avoid unnecessary tensor overhead for functions not implemented in NQCD
        dipole=pybuiltins.None, # Avoid unnecessary tensor overhead for functions not implemented in NQCD
        charges=pybuiltins.None, # Avoid unnecessary tensor overhead for functions not implemented in NQCD
        weight=Py(one(eltype(R))), # Can't avoid creating these tensors due to logic fallacy in MACE0.3.3
        energy_weight=Py(one(eltype(R))), # Can't avoid creating these tensors due to logic fallacy in MACE0.3.3
        forces_weight=Py(one(eltype(R))), # Can't avoid creating these tensors due to logic fallacy in MACE0.3.3
        stress_weight=Py(one(eltype(R))), # Can't avoid creating these tensors due to logic fallacy in MACE0.3.3
        virials_weight=Py(one(eltype(R))), # Can't avoid creating these tensors due to logic fallacy in MACE0.3.3
        config_type=Py("Default"),
        pbc=Py(pbc),
        cell=numpy[].array(cell_array),
    )
    return Py(config)
end

# ToDo: Evaluation function that handles model evaluation and caching of results.

"""
    predict!(
    mace_interface::MACEModel,
    atoms::Union{Vector{<:Atoms}, Atoms},
    R::Vector{<:AbstractMatrix},
    cell::Union{Vector{<:AbstractCell}, AbstractCell},
    )

Evaluate the MACE model on a set of structures if they haven't already been evaluated.
The results are stored in `mace_interface.last_eval_cache`.

Use `methodswith(MACEPredictionCache)` to see what data is provided from the results

# Arguments
`mace_interface`: `MACEModel` to evaluate with.

`atoms`: Atoms object for the structures passed for evaluation.
Can be either a vector of different Atoms objects or a single Atoms object.

`R`: Vector of structures to evaluate.

`cell`: Cell object for the structures passed for evaluation.
Can be either a vector of different Cell objects or a single Cell object.
"""
function predict!(
    mace_interface::MACEModel,
    atoms::Union{Vector{<:Atoms},Atoms},
    R::Vector{<:AbstractMatrix},
    cell::Union{Vector{<:AbstractCell},AbstractCell},
)
    if R != mace_interface.last_eval_cache.input_structures # Only predict if working on new structures
        dataset = Vector{Any}(undef, length(R))
        isa(cell, AbstractCell) ? cell = [cell for _ in 1:length(R)] : nothing # Always have atoms, positions and cell for each structure
        isa(atoms, Atoms) ? atoms = [atoms for _ in 1:length(R)] : nothing
        for i in eachindex(R)
            config = mace_configuration_from_nqcd_configuration(atoms[i], cell[i], R[i]; dtype=mace_interface.default_dtype)
            dataset[i] = mace_data[].AtomicData.from_config(config, mace_interface.z_table, mace_interface.cutoff_radius)
        end
        # Initialise DataLoader
        batch_size = mace_interface.batch_size === nothing ? length(dataset) : mace_interface.batch_size # Ensure there is a batch size
        mace_DataLoader = mace_tools[].torch_geometric.dataloader.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=false,
            drop_last=false,
        )
        # Allocate results arrays for prediction
        mace_interface.last_eval_cache.input_structures = deepcopy(R)
        mace_interface.last_eval_cache.energies = [zeros(mace_interface.default_dtype, length(mace_interface.models)) for i in eachindex(R)]
        mace_interface.last_eval_cache.forces = [zeros(mace_interface.default_dtype, mace_interface.ndofs, length(atoms[i]), length(mace_interface.models)) for i in eachindex(R)]


        # Iterate through dataloader and evaluate each model
        for (batch_index, batch) in enumerate(mace_DataLoader)
            evalcache_index = (batch_index - 1) * batch_size # Pointer to the start of the batch in the output arrays
            for (model_index, model) in enumerate(mace_interface.models)
                # Place copy of batch on model device
                clone = batch.clone().to(mace_interface.device[model_index])
                # Evaluate model
                model_output = model(clone.to_dict(), compute_stress=true)
                # Split according to batching
                #! Check how well this performs and whether this actually saves memory
                energies = Array(from_dlpack(model_output["energy"].detach()))
                forces = Array(from_dlpack(model_output["forces"].detach()))
                splitting = Array(from_dlpack(clone.ptr)) .+ 1 # Array of batch item bounds in output arrays, +1 due to Julia-Python conversion
                for structure_index in 2:length(splitting)
                    mace_interface.last_eval_cache.energies[evalcache_index+structure_index-1][model_index] = energies[structure_index-1]
                    mace_interface.last_eval_cache.forces[evalcache_index+structure_index-1][:, :, model_index] .= forces[:, splitting[structure_index-1]:splitting[structure_index]-1] # last index -1 because Julia includes last index in a slice
                end
            end
        end
    end
end

"""
    predict(
    mace_interface::MACEModel,
    atoms::Union{Vector{<:Atoms}, Atoms},
    R::Vector{<:AbstractMatrix},
    cell::Union{Vector{<:AbstractCell}, AbstractCell},
    )

Evaluate the MACE model on a set of structures if they haven't already been evaluated.
The results are returned as a `MACEPredictionCache`.

Use `methodswith(MACEPredictionCache)` to see what data can be provided from the results.

# Arguments
`mace_interface`: `MACEModel` to evaluate with.

`atoms`: Atoms object for the structures passed for evaluation.
Can be either a vector of different Atoms objects or a single Atoms object.

`R`: Vector of structures to evaluate.

`cell`: Cell object for the structures passed for evaluation.
Can be either a vector of different Cell objects or a single Cell object.
"""
function predict(
    mace_interface::MACEModel,
    atoms::Union{Vector{<:Atoms},Atoms},
    R::Vector{<:AbstractMatrix},
    cell::Union{Vector{<:AbstractCell},AbstractCell},
)
    predict!(mace_interface, atoms, R, cell)
    return deepcopy(mace_interface.last_eval_cache)
end

# Methods using MACEPredictionCache that convert MACE outputs to NQCD's atomic unit scheme. Snip length 1 caches to the basic outputs instead of unnecessary vector wrapping.

"""
    get_energy_mean(mace_cache::MACEPredictionCache)

Returns the mean potential energy of the structures stored in the evaluation cache.

Energy is returned in units of **Hartree**.
"""
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

"""
    get_energy_variance(mace_cache::MACEPredictionCache)

Returns the variance of the potential energy in **Hartree²**.
"""
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

"""
    get_energy_ensemble(mace_cache::MACEPredictionCache)

Returns the potential energy evaluated by each model in the ensemble in units of **Hartree**.
"""
function get_energy_ensemble(mace_cache::MACEPredictionCache)
    ensemble_energies = Vector{typeof(mace_cache.energies[1])}(undef, length(mace_cache.energies))
    for index in eachindex(mace_cache.energies)
        ensemble_energies[index] = @. austrip(mace_cache.energies[index] * u"eV") # Energy is given in eV
    end
    if length(ensemble_energies) == 1
        return ensemble_energies[1]
    else
        return ensemble_energies # Return in Hartree
    end
end

"""
    get_forces_mean(mace_cache::MACEPredictionCache)

Returns the mean forces of the structures stored in the evaluation cache.
Forces are returned in units of **Hartree/Bohr**.
"""
function get_forces_mean(mace_cache::MACEPredictionCache)
    mean_forces = Vector{Matrix{eltype(mace_cache.forces[1])}}(undef, length(mace_cache.forces))
    for index in eachindex(mace_cache.forces)
        mean_forces[index] = dropdims(mean(austrip.(mace_cache.forces[index] .* u"eV/Å"); dims=3); dims=3) # Force is given in eV/Å
    end
    if length(mean_forces) == 1
        return mean_forces[1]
    else
        return mean_forces # Return in Hartree / Bohr
    end
end

"""
    get_forces_variance(mace_cache::MACEPredictionCache)

Returns the force variance of the structures stored in the evaluation cache.
Forces are returned in units of **Hartree²/Bohr²**.
"""
function get_forces_variance(mace_cache::MACEPredictionCache)
    mean_forces = Vector{Matrix{eltype(mace_cache.forces[1])}}(undef, length(mace_cache.forces))
    for index in eachindex(mace_cache.forces)
        mean_forces[index] = dropdims(var(austrip.(mace_cache.forces[index] .* u"eV/Å"); dims=3); dims=3) # Force is given in eV/Å
    end
    if length(mean_forces) == 1
        return mean_forces[1]
    else
        return mean_forces # Return in Hartree / Bohr^2
    end
end

"""
    get_forces_ensemble(mace_cache::MACEPredictionCache)

Returns the forces evaluated by each model in the ensemble in units of **Hartree/Bohr**.
"""
function get_forces_ensemble(mace_cache::MACEPredictionCache)
    ensemble_forces = Vector{typeof(mace_cache.forces[1])}(undef, length(mace_cache.forces))
    for index in eachindex(mace_cache.forces)
        ensemble_forces[index] = permutedims(austrip.(mace_cache.forces[index] .* u"eV/Å"), (2, 1, 3)) # Force is given in eV/Å
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


function NQCModels.derivative(model::MACEModel, atoms::Atoms, R::AbstractMatrix, cell::Union{InfiniteCell,PeriodicCell})
    # Evaluate model
    D = zeros(eltype(R), size(R))
    predict!(model, atoms, [R], cell)
    # Return derivative (mean is trivial)
    @views D[:, model.mobile_atoms] .= -get_forces_mean(model.last_eval_cache)[:, model.mobile_atoms]
    return D
end

function NQCModels.derivative!(model::MACEModel, D::AbstractMatrix, atoms::Atoms, R::AbstractMatrix, cell::Union{InfiniteCell,PeriodicCell})
    # Evaluate model
    predict!(model, atoms, [R], cell)
    # Return derivative
    @views D[:, model.mobile_atoms] .-= get_forces_mean(model.last_eval_cache)[:, model.mobile_atoms]
end

# ToDo: Potential and derivative for multiple structures
"""
    NQCModels.potential(model::MACEModel, atoms::Atoms, R::Vector{<:AbstractMatrix}, cell::AbstractCell)

This variant of `NQCModels.potential` can make use of batch evaluation to speed up
inference for multiple structures.
"""
function NQCModels.potential(model::MACEModel, atoms::Atoms, R::Vector{<:AbstractMatrix}, cell::AbstractCell)
    # Evaluate model
    predict!(model, atoms, R, cell)
    # Return potential (mean is trivial here)
    return get_energy_mean(model.last_eval_cache)
end

"""
    NQCModels.derivative(model::MACEModel, atoms::Atoms, R::Vector{<:AbstractMatrix}, cell::Union{InfiniteCell, PeriodicCell})

This variant of `NQCModels.derivative` can make use of batch evaluation to speed up
inference for multiple structures.
"""
function NQCModels.derivative(model::MACEModel, atoms::Atoms, R::Vector{<:AbstractMatrix}, cell::Union{InfiniteCell,PeriodicCell})
    # Evaluate model
    D = zeros(eltype(R), size(R))
    predict!(model, atoms, R, cell)
    # Return derivative (mean is trivial)
    D_full = get_forces_mean(model.last_eval_cache)
    for i in axes(D, 3)
        @views D[i][:, model.mobile_atoms] .= -D_full[i][:, model.mobile_atoms]
    end
    return D
end

"""
    NQCModels.derivative(model::MACEModel, atoms::Atoms, R::AbstractArray{T,3}, cell::Union{InfiniteCell, PeriodicCell})

This variant of `NQCModels.derivative` can make use of batch evaluation to speed up
inference for multiple structures.
"""
function NQCModels.derivative!(model::MACEModel, atoms::Atoms, D::AbstractArray{T,3}, R::Vector{<:AbstractMatrix}, cell::Union{InfiniteCell,PeriodicCell}) where {T}
    # Evaluate model
    predict!(model, atoms, R, cell)
    # Return derivative (mean is trivial)
    for i in axes(D, 3)
        @views D[:, model.mobile_atoms, i] .-= get_forces_mean(model.last_eval_cache)[i][:, model.mobile_atoms]
    end
end

export predict, predict!, get_energy_mean, get_energy_variance, get_energy_ensemble, get_forces_mean, get_forces_variance, get_forces_ensemble, MACEModel, MACEPredictionCache
end
