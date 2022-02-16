using Libdl
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

struct AdiabaticEMTModel{A,F} <: AdiabaticModel
    atoms#::A
    cell#::PeriodicCell
    wrapper_function#::F
    pes_path#::AbstractString
    function AdiabaticEMTModel(atoms, cell, lib_path, pes_path)
        library = Libdl.dlopen(lib_path)
        #library = Libdl.dlopen("/home/chem/msrvhs/git_repos/md_tian2/src/md_tian2_lib.so")
        wrapper = Libdl.dlsym(library, :force_mp_full_energy_force_wrapper_)
        new{typeof(atoms),typeof(wrapper)}(atoms, cell, wrapper, pes_path)
    end
end

NQCModels.ndofs(::AdiabaticEMTModel) = 3


# function AdiabaticEMTModel(atoms, cell, lib_path, pes_path)
#     library = Libdl.dlopen(lib_path)
#     #library = Libdl.dlopen("/home/chem/msrvhs/git_repos/md_tian2/src/md_tian2_lib.so")
#     wrapper = Libdl.dlsym(library, :force_mp_full_energy_force_wrapper_)
#     return AdiabaticEMTModel(atoms, cell, wrapper, pes_path)
# end
    
function NQCModels.potential(model::AdiabaticEMTModel, R::AbstractMatrix)
    # set_coordinates!(model, R)

    natoms = size(model.atoms.types)[1]
    nbeads = 1 #Kept for future use?
    ntypes = 2 #Projectile and lattice

    cell_array = model.cell.vectors

    r = zeros(3,nbeads,natoms) 
    r[:,1,:] = R[:,:]
    f = zeros(3,nbeads,natoms) #forces
    V = [Float64(0.)] #size nbeads

    natoms_list = zeros(Int32,ntypes)
    natoms_list[1] = 1 #one projectile
    natoms_list[2] = natoms - 1 # n-1 lattice atoms

    # Assume one projectile that is atom 1
    # Assume one lattice with same elements
    projectile_element = Vector{UInt8}(string(model.atoms.types[1])) #code wants this to check pes file is right
    surface_element = Vector{UInt8}(string(model.atoms.types[2]))
    is_proj = [true,false]


    pes_file = Vector{UInt8}(model.pes_path)

    @assert natoms == sum(natoms_list)
    @assert length(is_proj) == ntypes


    ccall(model.wrapper_function,
            Cvoid,

            (Ref{Int32},Ref{Int32},Ref{Int32},Ptr{Float64},Ptr{Float64},
            Ptr{Float64},Ptr{Float64},Ptr{UInt8},Ref{Int},
            Ptr{Int32},Ptr{UInt8},Ptr{UInt8},Ptr{BlasInt}),

            natoms, nbeads, ntypes, r, cell_array,
            f, V, pes_file, size(pes_file), 
            natoms_list, projectile_element, surface_element, is_proj)


    return austrip(V[1] * u"eV")
end

function NQCModels.derivative!(model::AdiabaticEMTModel, R::AbstractMatrix)

    natoms = size(model.atoms.types)[1]
    nbeads = 1 #Kept for future use?
    ntypes = 2 #Projectile and lattice

    cell_array = model.cell.vectors

    r = zeros(3,nbeads,natoms) 
    r[:,1,:] = R[:,:]
    f = zeros(3,nbeads,natoms) #forces
    V = [Float64(0.)] #size nbeads

    natoms_list = zeros(Int32,ntypes)
    natoms_list[1] = 1 #one projectile
    natoms_list[2] = natoms - 1 # n-1 lattice atoms

    # Assume one projectile that is atom 1
    # Assume one lattice with same elements
    projectile_element = Vector{UInt8}(string(model.atoms.types[1])) #code wants this to check pes file is right
    surface_element = Vector{UInt8}(string(model.atoms.types[2]))
    is_proj = [true,false]


    pes_file = Vector{UInt8}(model.pes_path)

    @assert natoms == sum(natoms_list)
    @assert length(is_proj) == ntypes


    ccall(model.wrapper_function,
            Cvoid,

            (Ref{Int32},Ref{Int32},Ref{Int32},Ptr{Float64},Ptr{Float64},
            Ptr{Float64},Ptr{Float64},Ptr{UInt8},Ref{Int},
            Ptr{Int32},Ptr{UInt8},Ptr{UInt8},Ptr{BlasInt}),

            natoms, nbeads, ntypes, r, cell_array,
            f, V, pes_file, size(pes_file), 
            natoms_list, projectile_element, surface_element, is_proj)


    D = f[:,1,:]
    #D .= -model.atoms.get_forces()'
    @. D = austrip(D * u"eV/Å")

    return D
end

# function set_coordinates!(model::AdiabaticEMTModel, R)
#     model.atoms.set_positions(ustrip.(auconvert.(u"Å", R')))
# end
