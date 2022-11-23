using Libdl: Libdl
using LinearAlgebra: BlasInt 

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
end

function md_tian2_EMT(atoms, cell, lib_path, pes_path)
    library = Libdl.dlopen(lib_path)
    pes_init = Libdl.dlsym(library, :wrapper_mp_wrapper_read_pes_)
    wrapper =  Libdl.dlsym(library, :wrapper_mp_wrapper_energy_force_)
    
    # Initialize pes
    natoms = size(atoms.types)[1]
    nbeads = 1 #Kept for future use?
    ntypes = 2 #Projectile and lattice

    # Md_tian2 seems to use row_major order (same as printed in POSCAR file)
    cell_array = transpose(cell.vectors)
    for i in 1:3
        for j in 1:3
            cell_array[i,j] = austrip(cell_array[i,j]/u"Å")
        end
    end
    
    natoms_list = zeros(Int32,ntypes)
    natoms_list[1] = 1 #one projectile
    natoms_list[2] = natoms - 1 # n-1 lattice atoms

    # Assume one projectile that is atom 1
    # Assume one lattice with same elements
    projectile_element = Vector{UInt8}(string(atoms.types[1])) #code wants this to check pes file is right
    surface_element = Vector{UInt8}(string(atoms.types[2]))
    is_proj = [1, 0]

    pes_file = Vector{UInt8}(pes_path)

    @assert natoms == sum(natoms_list)
    @assert length(is_proj) == ntypes

    ccall(pes_init,
        Cvoid,

        (Ref{Int32},Ref{Int32},Ref{Int32},Ref{Float64},
        Ref{UInt8},Ref{Int},
        Ref{Int32},Ref{UInt8},Ref{UInt8},Ref{BlasInt}),

        natoms, nbeads, ntypes, cell_array,
        pes_file, size(pes_file), 
        natoms_list, projectile_element, surface_element, is_proj
    )

    r = zeros(3,nbeads,natoms) #initialize position array to pass to md_tian2
    f = zeros(3,nbeads,natoms) #initialize force array to pass to md_tian2

    V = zeros(Float64, nbeads)

    md_tian2_EMT(wrapper, library, r, f, natoms, nbeads, V)
end

NQCModels.ndofs(::md_tian2_EMT) = 3

function NQCModels.potential(model::md_tian2_EMT, R::AbstractMatrix)
    set_coordinates!(model, R)
    fill!(model.f, zero(eltype(model.f)))

    positions = zeros(size(model.r)[1],size(model.r)[2])
    for i in 1:size(model.r)[1]
        for j in 1:size(model.r)[2]
            positions[i,j] = austrip(positions[i,j]/u"Å")
        end
    end

    ccall(model.wrapper_function,
        Cvoid,

        (Ref{Int32}, Ref{Int32},
         Ref{Float64}, Ref{Float64}, Ref{Float64}),

        model.natoms, model.nbeads,
        # model.r, model.f, model.V
        positions, model.f, model.V
    )

    return austrip(model.V[1] * u"eV")
end

function NQCModels.derivative!(model::md_tian2_EMT, D::AbstractMatrix, R::AbstractMatrix)

    set_coordinates!(model, R)
    fill!(model.f, zero(eltype(model.f)))

    positions = zeros(size(model.r)[1],size(model.r)[2])
    for i in 1:size(model.r)[1]
        for j in 1:size(model.r)[2]
            positions[i,j] = austrip(positions[i,j]/u"Å")
        end
    end

    ccall(model.wrapper_function,
        Cvoid,

        (Ref{Int32},Ref{Int32},
         Ref{Float64},Ref{Float64},Ref{Float64}),

        model.natoms, model.nbeads,
        positions, model.f, model.V
    )

    for i in 1:model.natoms
        for j in 1:3
            D[j,i] = -austrip(model.f[j,1,i] * u"eV/Å")
        end
    end

    return D
end

function set_coordinates!(model::md_tian2_EMT, R)
    for i in 1:model.natoms
        for j in 1:3
            model.r[j,1,i] = ustrip(auconvert(u"Å", R[j,i]))
        end
    end
end

