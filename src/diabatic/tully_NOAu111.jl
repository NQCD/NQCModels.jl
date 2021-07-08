export Tully_NOAu111
#export NOAuVariables

@doc raw"""
    Tully's NO on Au111 potential energy surface, kindly shared with us by
    Sascha Kandratsenka and Belal
    See: RoyShenviTully_JChemPhys_130_174716_2009
"""
@with_kw struct Tully_NOAu111 <: LargeDiabaticModel
    m::Float64 = 2000
    ω::Float64 = 2e-4
    g::Float64 = 20.6097
    ΔG::Float64 = -3.8e-3
    Γ::Float64 = 6.4e-3
    W::Float64 = 5Γ
    M::Int = 40
    ρ::Float64 = M/2W
    increment::Float64 = 1/ρ
    n_states::Int = M+1
    n_au::Int=36
    au_pos::Matrix = zeros(n_au,3)
    no_pos::Matrix = zeros(2,3)
    Δ_no::Vector = no_pos[1,:] .- no_pos[2,:]
    mass::Vector = [14.00307440, 15.99491502, 196.966548] #This needs to be converted into amu
    a_lat::Float64 = 4.172
    x_pos::Matrix = zeros(n_au+2,3)
end

# @with_kw struct Tully_NOAu111 <: LargeDiabaticModel
#     au_atoms::Vector{Float}
#     # x::Array{Float64, 2}
#     # v::Array{Float64, 2}
#     #Δ_no::MVector{3, Float64}
#     # nn_arr::Array{Int64, 2}
#     # dhp_neutral::Array{Float64, 2}
#     # dhp_ion::Array{Float64, 2}
#     # dhp_coup::Array{Float64, 2}
#     # hp::MVector{3, Float64}
#     # F::Array{Float64, 2}
#     # surfp::MVector{A, Int16}
#     # occnum::MVector{C, Int16}
#     # surfh::MVector{D, Int16}
#     # surfpinit::MVector{A, Int16}
#     # surfpnew::MVector{A, Int16}
#     # trajzmin::MVector{1, Float64}
#     # trajtheta::MVector{1, Float64}
#     # H::Array{Float64, 2}
#     # ψ::Array{ComplexF64, 2}
#     # ϕ::Array{ComplexF64, 2}
#     # Γ::Array{Float64, 2}
#     # λ::MVector{C, Float64}
#     # dhdea::Array{Float64, 2}
#     # dhdv::Array{Float64, 2}
#     # akl::MVector{1, ComplexF64}
#     # akk::MVector{1, Float64}
#     # dm::Array{Float64, 2}
#     # blk::Array{Float64, 2}
#     # blk2::Array{Float64, 2}
#     # Pb::Array{Float64, 1}
#     # storage_aop::Array{Float64, 1}
#     # storage_op::Array{Float64, 2}
#     # storage_e::Array{Float64, 2}
#     # storage_xno::Array{Float64, 2}
#     # storage_vno::Array{Float64, 2}
#     # storage_temp::Array{Float64, 1}
#     # storage_phop::Array{Float64, 1}
#     # phipsi::MVector{1, ComplexF64}
#     # Pbmaxest::MVector{1, Float64}
#     # vdot::Array{Float64, 2}
#     # vtemp::Array{Float64, 2}
#     # vscale::MVector{1, Float64}
#     # storage_K::Array{Float64, 1}
#     # storage_P::Array{Float64, 1}
#     # storage_state::Array{Float64, 1}
#     # storage_deltaKNO::Array{Float64, 1}
#     # storage_deltaKAu::Array{Float64, 1}
#     # storage_hoptimes::Array{Int64, 1}
#     # exnum::MVector{1, Int64}
#     # attnum::MVector{1, Int64}
#     # nf::MVector{1, Int64}
#     # function Simulation_NOAu(n_au::Integer)
#     #     au_atoms = zeros(n_au,3)
#     #     new(au_atoms)
#     # end
# end

# function NOAuVariables(pos::AbstractArray)
#     au_atoms = pos
# end


function potential!(model::Tully_NOAu111, V::Hermitian, R::AbstractMatrix)
    @unpack m, ω, g, ΔG, Γ, ρ, increment, n_states, n_au, au_pos, no_pos, 
    Δ_no, mass, a_lat, x_pos  = model

    #Load in potential parameters
    param = PES_param(a_lat)
    
    #set up parameters like position arrays
    x_pos = [no_pos; au_pos]

    

    # Get info for neutral diabate
    E_au_au = get_V_au_au(model, param)

    E_n_o_neutral = get_E_n_o_neutral(model,param)
    
    V.data[1,1] = E_n_o_neutral

    return V
end

function get_E_n_o_neutral(s::Tully_NOAu111, p) #its good
    norm_r_no = norm(s.Δ_no)
    return p.F_n * (1 - exp(-p.δ_n*(norm_r_no - p.r_0_NO)))^2
end

#---------------------------GOLD AU AU LATTICE----------------------------------
# @inline function V_au_au_loop_if(x::float_array, V::Float64, i::Int, j::Int,
#     xm, xi, temp,r, qf_1, qf_2, s::Simulation)::Float64
#    if s.nn_arr[i, j] != 0
#        m = s.nn_arr[i, j] + 2
#        @views xm .= x[m, :]
#        @views xi .= x[i+2, :]
#        temp .= xm .- xi
#        temp .= temp./cell
#        temp .= temp .- floor.(temp .+ 0.5)
#        @views r .=  temp .* cell .- r0[:, j]
#        if j == 1 || j== 7
#            mul!(qf_2, d1_new, r)
#            qf_1 = dot(r, qf_2)
#            V = V + qf_1
#        elseif j == 2 || j== 8
#            mul!(qf_2, d2_new, r)
#            qf_1 = dot(r, qf_2)
#            V = V + qf_1
#        elseif j == 3 || j== 9
#            mul!(qf_2, d3_new, r)
#            qf_1 = dot(r, qf_2)
#            V = V + qf_1
#        elseif j == 4 || j== 10
#            mul!(qf_2, d4_new, r)
#            qf_1 = dot(r, qf_2)
#            V = V + qf_1
#        elseif j == 5|| j== 11
#            mul!(qf_2, d5_new, r)
#            qf_1 = dot(r, qf_2)
#            V = V + qf_1
#        elseif j == 6 || j== 12
#            mul!(qf_2, d6_new, r)
#            qf_1 = dot(r, qf_2)
#            V = V + qf_1
#        end
#    end
#    return V
# end
# @inline @inbounds function V_au_au_loop(x::float_array, V::Float64, s::Simulation, p)
#    xm = zeros(Float64, 3)
#    xi = zeros(Float64, 3)
#    temp = zeros(Float64, 3)
#    r = zeros(Float64, 3)
#    qf_1 = 0.0
#    qf_2 = zeros(Float64, 3)
#    for i in 1:N
#        for j in 1:12
#            V = V_au_au_loop_if(x, V, i, j, xm, xi, temp,r, qf_1, qf_2, s)
#        end
#    end
#    return V
# end


function get_V_au_au(s::Tully_NOAu111, p)    #check
    V = 0.0
    #V = V_au_au_loop(s.x_pos, V, s, p)
    return V/4.0
end

include("Tully_NOAu111/setup_parameters.jl")