import .JuLIP

using NonadiabaticDynamicsBase: PeriodicCell, Atoms
using NonadiabaticDynamicsBase: au_to_u, au_to_ang, eV_to_au, eV_per_ang_to_au


export JuLIPModel

"""
    JuLIPModel(atoms::Atoms{N,T}, cell::PeriodicCell,
               calculator::JuLIP.AbstractCalculator) where {N,T}

Model for interfacing with JuLIP potentials.
"""
mutable struct JuLIPModel{T,A<:NamedTuple,B<:NamedTuple} <: AdiabaticModel
    atoms::JuLIP.Atoms{T}
    tmp::A
    tmp_d::B
    function JuLIPModel(atoms::Atoms{N,T}, cell::PeriodicCell,
                        calculator::JuLIP.AbstractCalculator) where {N,T}

        jatoms = JuLIP.Atoms{T}(;
            X=zeros(3,length(atoms)),
            P=zeros(3,length(atoms)),
            M=au_to_u.(atoms.masses),
            Z=JuLIP.AtomicNumber.(atoms.numbers),
            cell=au_to_ang.(cell.vectors'),
            pbc=cell.periodicity,
            calc=calculator)

        tmp = JuLIP.alloc_temp(calculator, jatoms)
        tmp_d = JuLIP.alloc_temp_d(calculator, jatoms)

        new{T,typeof(tmp),typeof(tmp_d)}(jatoms, tmp, tmp_d)
    end
end

NonadiabaticModels.ndofs(::JuLIPModel) = 3

function NonadiabaticModels.potential(model::JuLIPModel, R::AbstractMatrix)
    JuLIP.set_positions!(model.atoms, au_to_ang.(R))
    try
        V = JuLIP.energy!(model.tmp, model.atoms.calc, model.atoms)
        return eV_to_au(V)
    catch e
        if e isa BoundsError
            model.tmp = JuLIP.alloc_temp(model.atoms.calc, model.atoms)
            V = JuLIP.energy!(model.tmp, model.atoms.calc, model.atoms)
            return eV_to_au(V)
        else
            throw(e)
        end
    end
end

function NonadiabaticModels.derivative!(model::JuLIPModel, D::AbstractMatrix, R::AbstractMatrix)
    JuLIP.set_positions!(model.atoms, au_to_ang.(R))
    try
        JuLIP.forces!(JuLIP.vecs(D), model.tmp_d, model.atoms.calc, model.atoms)
    catch e
        if e isa BoundsError
            model.tmp_d = JuLIP.alloc_temp_d(model.atoms.calc, model.atoms)
            JuLIP.forces!(JuLIP.vecs(D), model.tmp_d, model.atoms.calc, model.atoms)
        end
    end
    D .= -eV_per_ang_to_au.(D)
end
