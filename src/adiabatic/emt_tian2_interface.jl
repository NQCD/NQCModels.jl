
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
struct AdiabaticEMTModel{A} <: AdiabaticModel
atoms::A
cell::PeriodicCell
end

NQCModels.ndofs(::AdiabaticEMTModel) = 3

function NQCModels.potential(model::AdiabaticEMTModel, R::AbstractMatrix)
set_coordinates!(model, R)

natoms = size(model.atoms.types)[1]
nbeads = 1 #Kept for future use?
ntypes = 2 #Projectile and lattice

r = zeros(3,nbeads,natoms) 
r[:,1,:] = R[:,:] #positions
simbox = cell #cell
f = zeros(3,nbeads,natoms) #forces
epot = [Float64(0.)] #size nbeads

lib_file = #location of md_tian2 shared library #WHERE?
pes_file = #location of chosen md_tian2 pes 
pes_file_length = size(pes_file) #I pass the length of the string because
# I don't know how to deal with assumed size arrays

natoms_list = zeros(Int32,ntypes)
natoms_list[1] = 1 #one projectile
natoms_list[2] = natoms - 1 # n-1 lattice atoms

# Assume one projectile that is atom 1
# Assume one lattice with same elements
projectile_element = Vector{UInt8}(model.atoms.types[1]) #code wants this to check pes file is right
surface_element = Vector{UInt8}(model.atoms.types[2])
is_proj = [true,false]

@assert natoms == sum(natoms_list)
@assert length(is_proj) == ntypes

# This returns forces and energies, can I both in one function? 
ccall((:force_mp_full_energy_and_force_wrapper_,lib_file),
        Cvoid,

        (Ref{Int32},Ref{Int32},Ref{Int32},Ptr{Float64},Ptr{Float64},
        Ptr{Float64},Ptr{Float64},Ptr{UInt8},Ref{Int},
        Ptr{Int32},Ptr{UInt8},Ptr{UInt8},Ptr{BlasInt}),

        natoms, nbeads, ntypes, r, simbox,
        f, epot, pes_file, pes_file_length, 
        natoms_list, projectile_element, surface_element, is_proj)


return austrip(V * u"eV")
end

# function NQCModels.derivative!(model::AdiabaticEMTModel, D::AbstractMatrix, R::AbstractMatrix)
# set_coordinates!(model, R)
# D .= -model.atoms.get_forces()'
# @. D = austrip(D * u"eV/Å")
# return D
# end

function set_coordinates!(model::AdiabaticEMTModel, R)
model.atoms.set_positions(ustrip.(auconvert.(u"Å", R')))
end
