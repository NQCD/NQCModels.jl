

abstract type Parameters
end

struct PES_param <: Parameters
# #iesh or electronic friction, set 0 for iesh and 1 for electronic friction
latticeopt::Integer
logopt::Integer
Å::Float64 # 1 Angström in meter
ev::Float64
F_n::Float64
δ_n::Float64
r_0_NO::Float64

    function PES_param(a_lat::Float64)
        kjpermol = 1.660538863e-21
        latticeopt = 0
        logopt = 1
        Å = 1.0e-10 # 1 Angström in meter
        ev = 1.60217646e-19 # 1 eV in Joule
        F_n = 638.5000000000*kjpermol #N--O: Morse: F
        δ_n=2.7430000000/Å #N--O: Morse: delta
        r_0_NO = 1.1507700000*Å#N--O: Morse: cutoff
        new(latticeopt, logopt, Å, ev, F_n, δ_n, r_0_NO)

    end
end
# global const amu = 1.66053886e-27	# 1 amu in kg
# global const hbar = 1.05457148e-34 #6.582119569e-16	# hbar in amu*angstrom^2/femtosecond^2*fs
# global const fs = 1.0e-15  	# fs in s

# global const kjpermol = 1.660538863e-21 # 1 kJ/mol in Joules
# global const N_a = 602214076000000000000000


# global const kb = 1.3806504e-23#0.08617*1e-3 * conv2 * conv3 	# boltzman constant [mev/Kelvin] in simulation units []

# #Simulation parameters
# const global Ne = 20     #number of electrons
# const global Ms = 40    #number of orbitals
# const global numtraj = 100   #number of trajectories
# const global tsteps = 100000
# const global thop = 1   #number of timesteps between surface hops
# const global twrite = 5     #number of timesteps between data writing
# const global dt = 0.1 * fs  #time stepsize in Femtoseconds
# #these energy parameters are used in subsequent calculations with other variables defined
# # in the simulation units Angstrom, amu, femtoseconds. Therefore we need to convert
# # kilojoule/mole to amu*angstrom^2/femtosecond^2
# const global r_no = 1.15077*Å		# Initial bond length of NO
# const global e_trans_i = 0.05*ev #translational energy in eV -> kilojoule/mole
# const global e_vib_i = 3.1*ev #vibrational energy in eV-> kilojoule/mole
# const global e_rot_i = 0.05*ev #translational energy in eV-> kilojoule/mole
# const global phi_inc = 0.0 #incident angle in radians
# const global tsurf = 300.0 #temperature of surface in Kelvin
# const global sim_time = tsteps * dt

# #

# Random.seed!(1234);

# #parameters of surface etc.
# const global delta_E = 7.0 * ev
# const global sqrt_de = sqrt(delta_E)
# const global z_end = 11.0 * Å




# #Parameters of Neutral Diabatic Matrix Element
# global const A_n = 457000.052788*kjpermol #Au--O: exponential repulsion: A
# global const α_n = 3.75257871821/Å #Au--O: exponential repulsion: alpha
# global const B_n = 30788.8486039*kjpermol #Au--N: exponential repulsion: B
# global const β_n = 2.9728905911/Å #Au--N: exponential repulsion: beta
# global const cutoff = 10.0*Å #Au--N: cutoff
# global const exp_beta_n_cutoff = exp(-β_n * cutoff)
# global const exp_alpha_n_cutoff = exp(-α_n * cutoff)
### global const F_n = 638.5000000000*kjpermol #N--O: Morse: F
### global const δ_n=2.7430000000/Å #N--O: Morse: delta
### global const r_0_NO = 1.1507700000*Å#N--O: Morse: cutoff

# #Parameters of IONIC Diabatic Matrix Element

# global const C_i = 1.25581276843*Å  # Image potential: C
# global const D_i = 347.2225355*kjpermol*Å  #Image potential: D
# global const z0_i = 1.153606314*Å #Image potential: z0
# global const A_i = 457000.052788*kjpermol  #Au--O: exponential repulsion: A
# global const α_i = 3.75257871821/Å  #Au--O: exponential repulsion: alpha
# global const B_i = 23.8597594272*kjpermol  #Au--N: Morse: B
# global const β_i=1.91014033785/Å  #Au--N: Morse: beta
# global const rN_sur_e_i = 2.38958832878*Å #Au--N: Morse: rN_sur_e
# global const F_i = 495.9809807*kjpermol   #N--O: Morse: F
# global const δ_i = 2.47093477934/Å  #N--O: Morse: delta
# global const rNO_e_i = 1.29289837288*Å  #N--O: Morse: cutoff
# global const KI = 512.06425722*kjpermol  #N--O: Morse: cutoff
# global const exp_alpha_i_cutoff = exp(-α_i * cutoff)
# global const exp_beta_i_cutoff1 = exp(-2*β_i * (cutoff - rN_sur_e_i))
# global const exp_beta_i_cutoff2 = exp(-β_i * (cutoff - rN_sur_e_i))
# # DEFINE PARAMETERS FOR COUPLING FUNCTION


# global const coup_a_N=-70.5259924491*kjpermol # Au--N: Exponential decay: a
# global const coup_b_N=0.00470023958504 # Au--N: Exponential decay: b
# global const coup_β_N=1.95982478112/Å # Au--N: Exponential decay: beta
# global const coup_a_O=-16.7488672932*kjpermol # Au--O: Exponential decay: a
# global const coup_b_O=0.00617151653727 # Au--O: Exponential decay: b
# global const coup_β_O=1.35353579356/Å  # Au--O: Exponential decay: beta
# global const Au_O_coupling_cutoff=10.0*Å # Au--O: Exponential decay: cutoff
# global const coup_cutoff_O = coup_a_O/(1+coup_b_O*exp(coup_β_O * Au_O_coupling_cutoff))
# global const coup_cutoff_N = coup_a_N/(1+coup_b_N*exp(coup_β_N * Au_O_coupling_cutoff))
# #Define parameters for Au-Au interaction potential


# #function to convert arrays to static/mutable staticarrays
# array_to_ma(x, N, c, L) = MMatrix{N, c, Float64, L}(x)
# array_to_sa(x, N, c, L) = SMatrix{N, c, Float64, L}(x)
# # in Newton/meters, multiply with conv1 so that resulting units of energy will
# # be in kilojoule per mole
# global const α = -4.94
# global const β = 17.15
# global const γ = 19.40

# u = permutedims([[-1.0 0.0 1.0]/sqrt(2.0);[1.0 -2.0 1.0]/sqrt(6.0);[-1.0 -1.0 -1.0]/sqrt(3.0)])
# global const U_sa = array_to_sa(u, 3, 3, 9)

# function def_d_matrices(α, β, γ)
#     #see paper from 1947 for definitions
#     d17 = [[α 0.0 0.0]; [0.0 β γ]; [0.0 γ β]]
#     d28 = [[α 0.0 0.0]; [0.0 β -γ]; [0.0 -γ β]]
#     d39 = [[β 0.0 γ]; [0.0 α 0.0]; [γ 0.0 β]]
#     d410 = [[β 0.0 -γ]; [0.0 α 0.0]; [-γ 0.0 β]]
#     d511 = [[β γ 0.0]; [γ β 0.0]; [0.0 0.0 α]]
#     d612 = [[β -γ 0.0]; [-γ β 0.0]; [0.0 0.0 α]]
#     d17 = array_to_sa(d17, 3, 3, 9)
#     d28 = array_to_sa(d28, 3, 3, 9)
#     d39 = array_to_sa(d39, 3, 3, 9)
#     d410 = array_to_sa(d410, 3, 3, 9)
#     d511 = array_to_sa(d511, 3, 3, 9)
#     d612 = array_to_sa(d612, 3, 3, 9)
#     return d17, d28, d39, d410, d511, d612
# end
# global const d17, d28, d39, d410, d511, d612 = def_d_matrices(α, β, γ)

# function compute_d_new_basis(d17, d28, d39, d410, d511, d612)
#     d1_new = U_sa' * d17 * U_sa
#     d2_new = U_sa' * (d28* U_sa)
#     d3_new = U_sa' * (d39* U_sa)
#     d4_new = U_sa'* (d410* U_sa)
#     d5_new = U_sa' * (d511* U_sa)
#     d6_new = U_sa' * (d612* U_sa)
#     return d1_new, d2_new, d3_new, d4_new, d5_new, d6_new
# end
# const global d1_new, d2_new, d3_new, d4_new, d5_new, d6_new = compute_d_new_basis(d17, d28, d39, d410, d511, d612)

# #set masses
# const global mass_arr =  SVector{3, Float64}(14.00307440, 15.99491502, 196.966548) * amu
# const global m_N = mass_arr[1]
# const global m_O = mass_arr[2]
# const global m_au = mass_arr[3]
# const global μ = m_N*m_O/(m_N + m_O)

# m_spread_1 =repeat(mass_arr, inner=[1, 3])
# m_spread_2 = repeat([m_au], inner=[1, 3], outer=[527, 1])
# const global m_spread = vcat(m_spread_1, m_spread_2)

# function get_r0()
#     r0_old_basis = 0.5*a_lat*collect([0,1,1,0,1,-1,1,0,1,-1,0,1 ,1,1,0 ,1,-1,0,0,-1,
#     -1,0,-1,1,-1,0,-1,1, 0,-1,-1,-1,0,-1,1,0])
#     r0_old_basis = reshape(r0_old_basis, (3,12))
#     r0_new_basis = zeros(Float64, 3, 12)
#     mul!(r0_new_basis, transpose(U_sa) ,r0_old_basis)
#     return r0_new_basis
# end

# global const r0 = array_to_sa(get_r0(), 3, 12, 3*12)

# function burkey_cantrell()
#     e_diabat = zeros(Float64, Ms)
#     h0 = zeros(Float64, Ms+1, Ms+1)
#     vm = zeros(Float64, Ms+1, Ms+1)
#     gauss = zeros(Float64, Int(Ms/2), Int(Ms/2))
#     for j in 1:(Int(Ms/2)-1)
#         gauss[j, j+1] = Float64(j)/sqrt((2.0*Float64(j) + 1.0)*(2.0*Float64(j) - 1.0))
#         gauss[j+1, j] = gauss[j, j+1]
#     end

#     eigvals_gauss = eigvals(gauss)
#     eigvecs_gauss = eigvecs(gauss)

#     for j in 2:(Int(Ms/2) + 1)
#         h0[j, j] = delta_E/4.0*eigvals_gauss[j-1] - delta_E/4.0
#         h0[j+Int(Ms/2), j+Int(Ms/2)] = delta_E/4.0*eigvals_gauss[j-1] + delta_E/4.0
#         e_diabat[j-1] = h0[j, j]
#         e_diabat[j + Int(Ms/2) - 1] = h0[j+Int(Ms/2), j+Int(Ms/2)]
#     end

#     vm[1, 2:Int(Ms/2) + 1] = @. sqrt(delta_E/4.0 * (2.0*eigvecs_gauss[1, :]^2))
#     vm[2:Int(Ms/2) + 1, 1] = @. sqrt(delta_E/4.0 * (2.0*eigvecs_gauss[1, :]^2))
#     vm[1, (2 + Int(Ms/2)):(Ms + 1)] = @. sqrt(delta_E/4.0 * (2.0*eigvecs_gauss[1, :]^2))
#     vm[(2 + Int(Ms/2)):(Ms + 1), 1] = @. sqrt(delta_E/4.0 * (2.0*eigvecs_gauss[1, :]^2))

#     return h0, e_diabat, vm
# end

# h0_temp, e_diabat_temp, vm_temp = burkey_cantrell()
# global const h0 = h0_temp
# global const e_diabat = SVector{Ms, Float64}(e_diabat_temp)
# global const vm = vm_temp


# dnint(x) = x <= zero(eltype(x)) ? floor(x - 0.5) : floor(x + 0.5)
# end 
