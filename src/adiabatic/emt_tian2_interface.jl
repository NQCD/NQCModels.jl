using Libdl: Libdl
using LinearAlgebra: BlasInt
using Unitful: ustrip
using UnitfulAtomic: auconvert
"""
AdiabaticASEModel{A} <: AdiabaticModel

Interface to Md_tian2 code (Molecular Dynamics Tian Xia 2)
Github link: https://github.com/akandra/md_tian2
Some modification to original code neccesary for the interface 

Specifically we are using md_tian2's interface to a number of EMT models that it
hosts.

# md_tian2 basic units:
#  Length : Ang
#  Time   : fs
#  Energy : eV

Implements both `potential` and `derivative!`.
"""
struct md_tian2_EMT{F,L} <: AdiabaticModel
    wrapper_function::F
    library::L # so we can shut it
    r::Array{Float64,3} #positions
    f::Array{Float64,3} #forces
    natoms::Int
    nbeads::Int
    V::Vector{Float64}
    freeze::Vector{Int}
    mobile_atoms::Vector{Int}
end


function md_tian2_EMT(atoms, cell, lib_path, pes_path; freeze=[])
    library = Libdl.dlopen(lib_path)
    pes_init = Libdl.dlsym(library, :wrapper_mp_wrapper_read_pes_)
    wrapper =  Libdl.dlsym(library, :wrapper_mp_wrapper_energy_force_)
    
    # Initialize pes
    natoms = length(atoms.types)
    nbeads = 1 #Kept for future use?
    mobile_atoms = setdiff(1:natoms, freeze)

    elements_symbols = unique(atoms.types) 
    ntypes= length(elements_symbols)

    elements = string.(unique(atoms.types))
    elements = Vector{UInt8}(join(elements,","))


    natoms_list = zeros(Int32,ntypes)
    for i in eachindex(elements_symbols)
        natoms_list[i] =  count(==(elements_symbols[i]),atoms.types)
    end

    # Md_tian2 seems to use row_major order (same as printed in POSCAR file)
    # cell_array = permutedims(cell.vectors,(2,1))
    cell_array = cell.vectors
    cell_array = auconvert.(u"Å", cell_array)/u"Å"

    is_proj = zeros(Int64,ntypes)
    is_proj[1] = 1 


    pes_file = Vector{UInt8}(pes_path)

    #Loics electronic density modification for O atom
    modification = :O in atoms.types


    @assert natoms == sum(natoms_list)
    @assert length(is_proj) == ntypes

    ccall(pes_init,
        Cvoid,

        (Ref{Int32},Ref{Int32},Ref{Int32},Ref{Float64},
        Ref{UInt8},Ref{Int},
        Ref{Int32},Ref{UInt8}, Ref{Int}, Ref{BlasInt},
        Ref{Float64}, 
        Ref{BlasInt}),

        natoms, nbeads, ntypes, cell_array,
        pes_file, length(pes_file), 
        natoms_list, elements, length(elements),  is_proj,
        ustrip.(auconvert.(u"eV*fs^2*Å^-2",atoms.masses)),
        modification

    )

    r = zeros(3,nbeads,natoms) #initialize position array to pass to md_tian2
    f = zeros(3,nbeads,natoms) #initialize force array to pass to md_tian2

    V = zeros(Float64, nbeads)

    md_tian2_EMT(wrapper, library, r, f, natoms, nbeads, V, freeze, mobile_atoms)
end

NQCModels.ndofs(::md_tian2_EMT) = 3
NQCModels.mobileatoms(model::md_tian2_EMT, ::Int) = model.mobile_atoms
NQCModels.mobileatoms(model::md_tian2_EMT) = model.mobile_atoms

function NQCModels.potential(model::md_tian2_EMT, R::AbstractMatrix)
    set_coordinates!(model, R)
    fill!(model.f, zero(eltype(model.f)))

    ccall(model.wrapper_function,
        Cvoid,

        (Ref{Int32}, Ref{Int32},
         Ref{Float64}, Ref{Float64}, Ref{Float64}),

        model.natoms, model.nbeads,
        model.r, model.f, model.V
    )

    return austrip(model.V[1] * u"eV")
end

function NQCModels.derivative!(model::md_tian2_EMT, D::AbstractMatrix, R::AbstractMatrix)

    set_coordinates!(model, R)
    fill!(model.f, zero(eltype(model.f)))

    ccall(model.wrapper_function,
        Cvoid,

        (Ref{Int32},Ref{Int32},
         Ref{Float64},Ref{Float64},Ref{Float64}),

        model.natoms, model.nbeads,
        model.r, model.f, model.V
    )

    @inbounds for i in NQCModels.mobileatoms(model)
        for j in 1:3
            D[j,i] = -austrip(model.f[j,1,i] * u"eV/Å")
        end
    end

    return D
end

function pseudo_ldfa!(model::md_tian2_EMT,d::Vector{Float64}, F::Vector{Float64}, R::AbstractMatrix)

    set_coordinates!(model, R)

    friction = zeros(1,length(F)) #nbeads, natoms
    density = zeros(1,length(d)) #nbeads, natoms

    friction_function =  Libdl.dlsym(model.library, :wrapper_mp_wrapper_dens_friction_)

    ccall(friction_function,
        Cvoid,

        (Ref{Int32},Ref{Int32},
         Ref{Float64},Ref{Float64},Ref{Float64}),

        model.natoms, model.nbeads,
        model.r, friction, density
    )

    @inbounds for i in NQCModels.mobileatoms(model)
        F[i] = austrip(friction[1,i] * u"fs^-1")
        d[i] = austrip(density[1,i] * u"Å^-3") 
    end



    return d,F
end

function set_coordinates!(model::md_tian2_EMT, R)
    for i in 1:model.natoms
        for j in 1:3
            model.r[j,1,i] = ustrip(auconvert(u"Å", R[j,i]))
        end
    end
end

# function find_layer_indices(r, layers)
#     freeze = Int[]
#     layers == 0 && return freeze

#     z_coordinates = @view r[3,:]
#     permutation = sortperm(z_coordinates)
#     ordered_z = z_coordinates[permutation]
#     current_z = ordered_z[begin]
#     current_layer = 1

#     for (i, z) in enumerate(ordered_z)
#         if !isapprox(z, current_z)
#             current_layer += 1
#             current_layer > layers && break
#             current_z = z
#         end
#         push!(freeze, permutation[i])
#     end
    
#     return freeze
# end

