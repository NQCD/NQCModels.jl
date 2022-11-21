using Libdl: Libdl
using NQCBase: PeriodicCell
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
struct md_tian2_EMT{A,F} <: AdiabaticModel
    atoms#::A
    cell#::PeriodicCell
    wrapper_function#::F
    pes_path#::AbstractString
    library #soo we can shut it
    r #positions
    f #forces
    natoms #number of atoms
    nbeads #number of beads
    V #potential energy
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
        natoms_list = zeros(Int32,ntypes)
        natoms_list[1] = 1 #one projectile
        natoms_list[2] = natoms - 1 # n-1 lattice atoms

        # Assume one projectile that is atom 1
        # Assume one lattice with same elements
        projectile_element = Vector{UInt8}(string(atoms.types[1])) #code wants this to check pes file is right
        surface_element = Vector{UInt8}(string(atoms.types[2]))
        is_proj = [true,false]

        pes_file = Vector{UInt8}(pes_path)

        @assert natoms == sum(natoms_list)
        @assert length(is_proj) == ntypes


        ccall(pes_init,
                Cvoid,

                (Ref{Int32},Ref{Int32},Ref{Int32},Ptr{Float64},
                Ptr{UInt8},Ref{Int},
                Ptr{Int32},Ptr{UInt8},Ptr{UInt8},Ptr{BlasInt}),

                natoms, nbeads, ntypes, cell_array,
                pes_file, size(pes_file), 
                natoms_list, projectile_element, surface_element, is_proj)


        r = zeros(3,nbeads,natoms) #initialize position array to pass to md_tian2
        f = zeros(3,nbeads,natoms) #initialize force array to pass to md_tian2

        V = [Float64(0.)] #size nbeads

        new{typeof(atoms),typeof(wrapper)}(atoms, cell, wrapper, pes_path, library, r, f, natoms, nbeads, V)
    end



end

NQCModels.ndofs(::md_tian2_EMT) = 3


# function md_tian2_EMT(atoms, cell, lib_path, pes_path)
#     library = Libdl.dlopen(lib_path)
#     #library = Libdl.dlopen("/home/chem/msrvhs/git_repos/md_tian2/src/md_tian2_lib.so")
#     wrapper = Libdl.dlsym(library, :force_mp_full_energy_force_wrapper_)
#     return md_tian2_EMT(atoms, cell, wrapper, pes_path)
# end

    
function NQCModels.potential(model::md_tian2_EMT, R::AbstractMatrix)
    # set_coordinates!(model, R)

    #natoms = size(model.atoms.types)[1]
    #nbeads = 1 #Kept for future use?
    #r = zeros(3,nbeads,natoms) 
    model.r[:,1,:] = R[:,:]
    model.f[:,1,:] .= 0.
    #f = zeros(3,nbeads,natoms) #forces
    #V = [Float64(0.)] #size nbeads

    # ccall(model.wrapper_function,
    #         Cvoid,

    #         (Ref{Int32},Ref{Int32},Ref{Int32},Ptr{Float64},Ptr{Float64},
    #         Ptr{Float64},Ptr{Float64},Ptr{UInt8},Ref{Int},
    #         Ptr{Int32},Ptr{UInt8},Ptr{UInt8},Ptr{BlasInt}),

    #         natoms, nbeads, ntypes, r, cell_array,
    #         f, V, pes_file, size(pes_file), 
    #         natoms_list, projectile_element, surface_element, is_proj)

    ccall(model.wrapper_function,
        Cvoid,

        (Ref{Int32},Ref{Int32},Ptr{Float64},
        Ptr{Float64},Ptr{Float64}),

        model.natoms, model.nbeads, model.r,
        model.f, model.V )

    return austrip(V[1] * u"eV")
end

function NQCModels.derivative!(model::md_tian2_EMT, D::AbstractMatrix, R::AbstractMatrix)

    #natoms = size(model.atoms.types)[1]
    #nbeads = 1 #Kept for future use?
    #r = zeros(3,nbeads,natoms) 
    #r[:,1,:] = R[:,:]
    #f = zeros(3,nbeads,natoms) #forces
    #V = [Float64(0.)] #size nbeads

    model.r[:,1,:] = R[:,:]
    model.f[:,1,:] .= 0.

    # ccall(model.wrapper_function,
    #         Cvoid,

    #         (Ref{Int32},Ref{Int32},Ref{Int32},Ptr{Float64},Ptr{Float64},
    #         Ptr{Float64},Ptr{Float64},Ptr{UInt8},Ref{Int},
    #         Ptr{Int32},Ptr{UInt8},Ptr{UInt8},Ptr{BlasInt}),

    #         natoms, nbeads, ntypes, r, cell_array,
    #         f, V, pes_file, size(pes_file), 
    #         natoms_list, projectile_element, surface_element, is_proj)

    ccall(model.wrapper_function,
        Cvoid,

        (Ref{Int32},Ref{Int32},Ptr{Float64},
        Ptr{Float64},Ptr{Float64}),

        model.natoms, model.nbeads, model.r,
        model.f, model.V )


    D .= model.f[:,1,:]
    #D .= -model.atoms.get_forces()'
    @. D = austrip(D * u"eV/Å")

    return D
end

# function NQCModels.cleanup(model::md_tian2_EMT)

#     deallocate= Libdl.dlsym(library, :wrapper_mp_wrapper_deallocations_)
#     ccall(deallocate,Cvoid,())
#     library = Libdl.dlclose(lib_path)

# end

# function set_coordinates!(model::md_tian2_EMT, R)
#     model.atoms.set_positions(ustrip.(auconvert.(u"Å", R')))
# end
