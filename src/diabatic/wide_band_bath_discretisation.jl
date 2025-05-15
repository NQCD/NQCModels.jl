using FastGaussQuadrature: gausslegendre

abstract type WideBandBathDiscretisation end
NQCModels.nstates(bath::WideBandBathDiscretisation) = length(bath.bathstates)

function fillbathstates!(out::Hermitian, bath::WideBandBathDiscretisation)
    diagonal = view(out, diagind(out)[2:end])
    copy!(diagonal, bath.bathstates)
end

function fillbathcoupling!(out::Hermitian, coupling::Real, bath::WideBandBathDiscretisation, couplings_rescale::Real=1.0)
    first_column = @view out.data[2:end, 1]
    setcoupling!(first_column, bath.bathcoupling, coupling, couplings_rescale)
    first_row = @view out.data[1, 2:end]
    copy!(first_row, first_column)
end

function setcoupling!(out::AbstractVector, bathcoupling::AbstractVector, coupling::Real, couplings_rescale::Real=1.0)
    out .= bathcoupling .* coupling .* couplings_rescale  # bath's states coupling (constant) * Hybridization coupling component V_k(r)
end

function setcoupling!(out::AbstractVector, bathcoupling::Real, coupling::Real, couplings_rescale::Real=1.0)
    fill!(out, bathcoupling * coupling * couplings_rescale)
end

"""
    TrapezoidalRule{B,T} <: WideBandBathDiscretisation

Discretise wide band continuum using trapezoidal rule.
Leads to evenly spaced states and constant coupling.
"""
struct TrapezoidalRule{B,T} <: WideBandBathDiscretisation
    bathstates::B   # ϵ
    bathcoupling::T # V(ϵ,x̃) 
end

function TrapezoidalRule(M, bandmin, bandmax; fermilevelstate = false)

    if fermilevelstate == false
        nothing
    elseif fermilevelstate == true
        M = M+1 # increase number of states in window region by 1 to place a state at the fermi level
        @info "Additional state added at Fermi level, number of states in discretisation =$M"
    else
        throw(error("Invalid value provided to `fermilevelstate`. Only Boolean values allowed."))
    end

    ΔE = bandmax - bandmin
    bathstates = range(bandmin, bandmax, length=M)
    bathcoupling = sqrt(ΔE / M)
    return TrapezoidalRule(bathstates, bathcoupling)
end

# ------------------------------------------------------------------------------------------------ #
#                                     New discretisation method                                    #
# ------------------------------------------------------------------------------------------------ #

"""
    WindowedTrapezoidalRule{B,T}

Discretise wide band continuum using the trapezoidal rule with a region of higher state density
surrounding the Fermi Energy. The position of the energy window containing more densely packed states
is defined by energy values and the increase in density by a proportion. 
"""

struct WindowedTrapezoidalRule{T} <: WideBandBathDiscretisation
    bathstates::Vector{T} # due to the non-even distribution need to represent as vectors
    bathcoupling::Vector{T} 
end


# function WindowScaling(bath_states, bath_couplings, window_energy,  slope)
#     M = length(bath_states)
#     M_fermi = floor(Int, M * 0.5) 

#     WindowCoupling = bath_couplings[M_fermi] 
#     ΔCoupling = bath_couplings[1] - WindowCoupling

#     L = W.bathstates[W.bathstates.≥-window_energy]
#     WindowRegion = L[L.≤window_energy]

#     function ScalingFn(x, a, b, c)
#         arg1 = b .* (x .- c)
#         arg2 = -b .* (x .+ c)
    
#         return a .* (exp.(arg1) + exp.(arg2)) .+ 1
#         # return (exp.(arg1) + exp.(arg2))
#     end    
    
#     WindScale = ScalingFn(WindowRegion, ΔCoupling, slope, window_energy)
#     return WindScale
# end

function ScalingFn(x, a, b, c)
    arg1 = b .* (x .- c)
    arg2 = -b .* (x .+ c)

    return a .* (exp.(arg1) + exp.(arg2)) .+ 1
    # return (exp.(arg1) + exp.(arg2))
end

# A = ScalingFn(WindowRegion, ΔCoupling, slope=50, window_energy=1.0)

"""
    WindowedTrapezoidalRule(M, bandmin, bandmax, windmin, windmax; densityratio=0.50)

Original discretisation method.
"""
function WindowedTrapezoidalRule(M, bandmin, bandmax, windmin, windmax; densityratio=0.50)
    (M - (M*densityratio)) % 2 == 0 || throw(error("For the provided arguments, the number of states not in the windowed region is $(M - (M*densityratio)). This value must be even for a symmetric discretisation.")) # constraint enforced such that the a densityratio=0.5 can be utilized and return an integer number of states.
    abs(bandmin) > abs(windmin) || throw(error("Requested window minimum energy lies outside of energy range."))
    abs(bandmax) > abs(windmax) || throw(error("Requested window maximum energy lies outside of energy range."))
    
    M_window = floor(Int, M*densityratio)
    M_sparse = floor(Int, (M - (M*densityratio))/2)
    
    # Sparsely distributed trapezoidal rule discretised bath states
    ΔE_sparse1 = windmin - bandmin # Energy range for first sparsely distributed state region
    bstates_sparse1 = collect(range(bandmin, windmin, length=M_sparse+1))[1:end-1] # slicing is implemented to rectify energy degeneracy at window bounds
    bcoupling_sparse1 = fill!(copy(bstates_sparse1), sqrt(ΔE_sparse1 / M_sparse))

    ΔE_sparse2 = bandmax - windmax # Energy Range for second sparsely distributed state region
    bstates_sparse2 = collect(range(windmax, bandmax, length=M_sparse+1))[2:end]
    bcoupling_sparse2 = fill!(copy(bstates_sparse2), sqrt(ΔE_sparse2 / M_sparse))


    # densely distributed trapezoidal rule discretised bath states within energy window
    ΔE_window = windmax - windmin
    bstates_window = collect(range(windmin, windmax, length=M_window))
    bcoupling_window = fill!(copy(bstates_window), sqrt(ΔE_window / M_window))


    bathstates = [bstates_sparse1; bstates_window; bstates_sparse2] # concatenates arrays vertically (along axis = 1)    
    bathcoupling = [bcoupling_sparse1; bcoupling_window; bcoupling_sparse2]

    return WindowedTrapezoidalRule(bathstates, bathcoupling)
end


"""
    WindowedTrapezoidalRule2(M, bandmin, bandmax, windmin, windmax; densityratio=0.50)

Adjusted method where the states at the window edges are set to have the same coupling value as the sparse region.
"""
function WindowedTrapezoidalRule2(M, bandmin, bandmax, windmin, windmax; densityratio=0.50)
    (M - (M*densityratio)) % 2 == 0 || throw(error("For the provided arguments, the number of states not in the windowed region is $(M - (M*densityratio)). This value must be even for a symmetric discretisation.")) # constraint enforced such that the a densityratio=0.5 can be utilized and return an integer number of states.
    abs(bandmin) > abs(windmin) || throw(error("Requested window minimum energy lies outside of energy range."))
    abs(bandmax) > abs(windmax) || throw(error("Requested window maximum energy lies outside of energy range."))
    
    M_window = floor(Int, M*densityratio)
    M_sparse = floor(Int, (M - (M*densityratio))/2)
    
    # Sparsely distributed trapezoidal rule discretised bath states
    ΔE_sparse1 = windmin - bandmin # Energy range for first sparsely distributed state region
    bstates_sparse1 = collect(range(bandmin, windmin, length=M_sparse+1)) # slicing is implemented to rectify energy degeneracy at window bounds
    bcoupling_sparse1 = fill!(copy(bstates_sparse1), sqrt(ΔE_sparse1 / M_sparse))

    ΔE_sparse2 = bandmax - windmax # Energy Range for second sparsely distributed state region
    bstates_sparse2 = collect(range(windmax, bandmax, length=M_sparse+1))
    bcoupling_sparse2 = fill!(copy(bstates_sparse2), sqrt(ΔE_sparse2 / M_sparse))


    # densely distributed trapezoidal rule discretised bath states within energy window
    ΔE_window = windmax - windmin
    bstates_window = collect(range(windmin, windmax, length=M_window))[2:end-1]
    bcoupling_window = fill!(copy(bstates_window), sqrt(ΔE_window / M_window))
    

    bathstates = [bstates_sparse1; bstates_window; bstates_sparse2] # concatenates arrays vertically (along axis = 1)    
    bathcoupling = [bcoupling_sparse1; bcoupling_window; bcoupling_sparse2]

    return WindowedTrapezoidalRule(bathstates, bathcoupling)
end

"""
    WindowedTrapezoidalRule3(M, bandmin, bandmax, windmin, windmax; densityratio=0.50, correctionslope=50)

Extension to method 2, with a `cosh(x)` type scaling correction applied to the coupling value in the window region.
"""
function WindowedTrapezoidalRule3(M, bandmin, bandmax, windmin, windmax; densityratio=0.50, correctionslope=50)
    (M - (M*densityratio)) % 2 == 0 || throw(error("For the provided arguments, the number of states not in the windowed region is $(M - (M*densityratio)). This value must be even for a symmetric discretisation.")) # constraint enforced such that the a densityratio=0.5 can be utilized and return an integer number of states.
    abs(bandmin) > abs(windmin) || throw(error("Requested window minimum energy lies outside of energy range."))
    abs(bandmax) > abs(windmax) || throw(error("Requested window maximum energy lies outside of energy range."))
    
    M_window = floor(Int, M*densityratio)
    M_sparse = floor(Int, (M - (M*densityratio))/2)
    
    # Sparsely distributed trapezoidal rule discretised bath states
    ΔE_sparse1 = windmin - bandmin # Energy range for first sparsely distributed state region
    bstates_sparse1 = collect(range(bandmin, windmin, length=M_sparse+1)) # slicing is implemented to rectify energy degeneracy at window bounds
    bcoupling_sparse1 = fill!(copy(bstates_sparse1), sqrt(ΔE_sparse1 / M_sparse))

    ΔE_sparse2 = bandmax - windmax # Energy Range for second sparsely distributed state region
    bstates_sparse2 = collect(range(windmax, bandmax, length=M_sparse+1))
    bcoupling_sparse2 = fill!(copy(bstates_sparse2), sqrt(ΔE_sparse2 / M_sparse))


    # densely distributed trapezoidal rule discretised bath states within energy window
    ΔE_window = windmax - windmin
    bstates_window = collect(range(windmin, windmax, length=M_window))[2:end-1]
    bcoupling_window = fill!(copy(bstates_window), sqrt(ΔE_window / M_window))

    ΔCoupling = sqrt(ΔE_sparse1 / M_sparse) - sqrt(ΔE_window / M_window)
    WindScale = ScalingFn(bstates_window, ΔCoupling, correctionslope, windmax)
    

    bathstates = [bstates_sparse1; bstates_window; bstates_sparse2] # concatenates arrays vertically (along axis = 1)    
    bathcoupling = [bcoupling_sparse1; bcoupling_window.*WindScale; bcoupling_sparse2]

    return WindowedTrapezoidalRule(bathstates, bathcoupling)
end

function CouplingCalc(x, scalingfactor, window, offset, gradient)
    fermi(ϵ, μ, β) = (exp(β*(ϵ - μ)) + 1)^(-1)  
    arg = scalingfactor .* fermi.(x, -window, gradient) .+ offset
    return [arg[1:50]; reverse(arg)[51:end]] 
    # return (exp.(arg1) + exp.(arg2))
end

"""
    WindowedTrapezoidalRule4(M, bandmin, bandmax, windmin, windmax; densityratio=0.50, correctionslope=50)

Extension to method 2, with a Fermi Dirac distribution type scaling correction applied to the coupling value around the window energy.
"""
function WindowedTrapezoidalRule4(M, bandmin, bandmax, windmin, windmax; densityratio=0.50, correctionslope=50)
    (M - (M*densityratio)) % 2 == 0 || throw(error("For the provided arguments, the number of states not in the windowed region is $(M - (M*densityratio)). This value must be even for a symmetric discretisation.")) # constraint enforced such that the a densityratio=0.5 can be utilized and return an integer number of states.
    abs(bandmin) > abs(windmin) || throw(error("Requested window minimum energy lies outside of energy range."))
    abs(bandmax) > abs(windmax) || throw(error("Requested window maximum energy lies outside of energy range."))
    
    M_window = floor(Int, M*densityratio)
    M_sparse = floor(Int, (M - (M*densityratio))/2)
    
    # Sparsely distributed trapezoidal rule discretised bath states
    ΔE_sparse1 = windmin - bandmin # Energy range for first sparsely distributed state region
    bstates_sparse1 = collect(range(bandmin, windmin, length=M_sparse+1)) # slicing is implemented to rectify energy degeneracy at window bounds
    bcoupling_sparse1 = sqrt(ΔE_sparse1 / M_sparse)

    ΔE_sparse2 = bandmax - windmax # Energy Range for second sparsely distributed state region
    bstates_sparse2 = collect(range(windmax, bandmax, length=M_sparse+1))
    bcoupling_sparse2 = sqrt(ΔE_sparse2 / M_sparse)

    bcoupling_sparse1 == bcoupling_sparse2 || throw(error("State distribution is not symmetric. The calculated couplings for both sparse regions do not match."))

    # densely distributed trapezoidal rule discretised bath states within energy window
    ΔE_window = windmax - windmin
    bstates_window = collect(range(windmin, windmax, length=M_window))[2:end-1]
    bcoupling_window = sqrt(ΔE_window / M_window)

    bathstates = [bstates_sparse1; bstates_window; bstates_sparse2]

    ΔCoupling = bcoupling_sparse1 - bcoupling_window
    bathcoupling = CouplingCalc(bathstates, ΔCoupling, windmax, bcoupling_window, correctionslope)

    return WindowedTrapezoidalRule(bathstates, bathcoupling)
end

"""
    WindowedTrapezoidalRule5(M, bandmin, bandmax, windmin, windmax; densityratio=0.50, correctionslope=50)

Extension to method 2, with arbitrary changes made to the coupling values that seem to work to generate a flat DOS - NEEDS REFINING!!!
"""
function WindowedTrapezoidalRule5(M, bandmin, bandmax, windmin, windmax; densityratio=0.50)
    (M - (M*densityratio)) % 2 == 0 || throw(error("For the provided arguments, the number of states not in the windowed region is $(M - (M*densityratio)). This value must be even for a symmetric discretisation.")) # constraint enforced such that the a densityratio=0.5 can be utilized and return an integer number of states.
    abs(bandmin) > abs(windmin) || throw(error("Requested window minimum energy lies outside of energy range."))
    abs(bandmax) > abs(windmax) || throw(error("Requested window maximum energy lies outside of energy range."))
    
    M_window = floor(Int, M*densityratio)
    M_sparse = floor(Int, (M - (M*densityratio))/2)
    
    # Sparsely distributed trapezoidal rule discretised bath states
    ΔE_sparse1 = windmin - bandmin # Energy range for first sparsely distributed state region
    bstates_sparse1 = collect(range(bandmin, windmin, length=M_sparse+1)) # slicing is implemented to rectify energy degeneracy at window bounds
    bcoupling_sparse1 = fill!(copy(bstates_sparse1), sqrt(ΔE_sparse1 / M_sparse))

    ΔE_sparse2 = bandmax - windmax # Energy Range for second sparsely distributed state region
    bstates_sparse2 = collect(range(windmax, bandmax, length=M_sparse+1))
    bcoupling_sparse2 = fill!(copy(bstates_sparse2), sqrt(ΔE_sparse2 / M_sparse))


    # densely distributed trapezoidal rule discretised bath states within energy window
    ΔE_window = windmax - windmin
    bstates_window = collect(range(windmin, windmax, length=M_window))[2:end-1]
    bcoupling_window = fill!(copy(bstates_window), sqrt(ΔE_window / M_window))
    bcoupling_window[1] *= 0.25
    bcoupling_window[end] *= 0.25
    bcoupling_window[2:end-1] *= 1.01  

    bathstates = [bstates_sparse1; bstates_window; bstates_sparse2] # concatenates arrays vertically (along axis = 1)    
    bathcoupling = [bcoupling_sparse1; bcoupling_window; bcoupling_sparse2]

    return WindowedTrapezoidalRule(bathstates, bathcoupling)
end


"""
    WindowedTrapezoidalRule6(M, bandmin, bandmax, windmin, windmax; densityratio=0.50)

Windowed discretisation where spacing increases linearly from window edge to bandwidth edge.                 
"""
function WindowedTrapezoidalRule6(M, bandmin, bandmax, windmin, windmax; densityratio=0.50, fermilevelstate=false)
    (M - (M*densityratio)) % 2 == 0 || throw(error("For the provided arguments, the number of states not in the windowed region is $(M - (M*densityratio)). This value must be even for a symmetric discretisation.")) # constraint enforced such that the a densityratio=0.5 can be utilized and return an integer number of states.
    abs(bandmin) > abs(windmin) || throw(error("Requested window minimum energy lies outside of energy range."))
    abs(bandmax) > abs(windmax) || throw(error("Requested window maximum energy lies outside of energy range."))
    
    M_window = floor(Int, M*densityratio)
    M_sparse = floor(Int, (M - (M*densityratio))/2)
    

    # densely distributed trapezoidal rule discretised bath states within energy window
    ΔE_window = windmax - windmin # Energy range for window region
    # bstates_window = collect(range(windmin, windmax, length=M_window))
    # bcoupling_window = fill!(copy(bstates_window), sqrt(ΔE_window / M_window))

    if fermilevelstate == false
        S_0 = ΔE_window/(M_window-1) # window spacing (for a N ticks there are N-1 spacings)
    elseif fermilevelstate == true
        S_0 = ΔE_window/M_window # window spacing
        M_window = M_window + 1
        M_new = M_window + (M_sparse * 2)
        @info "Additional state added at Fermi level, number of states in discretisation =$M_new"
    else
        throw(error("Invalid value provided to `fermilevelstate`. Only Boolean values allowed."))
    end

    bstates_window = collect(range(windmin, windmax, length=M_window))
    bcoupling_window = fill!(copy(bstates_window), sqrt(S_0))

    
    windowfraction = ΔE_window/(bandmax - bandmin)
    if windowfraction < densityratio
        @info("Input parameters define a window region denser than the outer region.")
    elseif windowfraction > densityratio
        @info("Input parameters define a window region sparser than the outer region.")
    elseif windowfraction == densityratio
        @info("Input parameters define a window region of equal density to the outer region.")
    else
        throw(error("Invalid parameters provided."))
    end


    # ΔE_sparse1 = windmin - bandmin # Energy range for first sparsely distributed state region
    ΔE_sparse2 = bandmax - windmax # Energy Range for second sparsely distributed state region


    S_max = (M_sparse*ΔE_sparse2 - M_sparse*(M_sparse-1)*S_0 + sum([i*S_0 for i in 1:(M_sparse-1)]))/(M_sparse + sum([i for i in 1:(M_sparse-1)]))
    
    S_max > 0 || throw(error("Too few states in window region, changing space in sparse region has extrapolated to a negative spacing"))
    
    gradient = (S_max - S_0)/M_sparse
    Spacing = [gradient*i + S_0 for i in 1:M_sparse]


    bstates_sparse = zeros(M_sparse)
    for (i, S_i) in enumerate(Spacing)
        if i == 1
            bstates_sparse[1] = windmax + S_i
        else
            bstates_sparse[i] = bstates_sparse[i-1] + S_i
        end
    end

    # Sparsely distributed trapezoidal rule discretised bath states
    # negative region
    bstates_sparse1 = -1 .* reverse(copy(bstates_sparse)) 
    bcoupling_sparse1 = sqrt.(reverse(copy(Spacing)))

    #positive region
    bstates_sparse2 = bstates_sparse
    bcoupling_sparse2 = sqrt.(Spacing)

    bathstates = [bstates_sparse1; bstates_window; bstates_sparse2] # concatenates arrays vertically (along axis = 1)    
    bathcoupling = [bcoupling_sparse1; bcoupling_window; bcoupling_sparse2]

    return WindowedTrapezoidalRule(bathstates, bathcoupling)
end


"""
    WindowedTrapezoidalRule7(M, bandmin, bandmax, windmin, windmax; densityratio=0.50)

Windowed discretisation where ShenviGaussLegendre discretisation is used outside the window region.                 
"""
function WindowedTrapezoidalRule7(M, bandmin, bandmax, windmin, windmax; densityratio=0.50, fermilevelstate=false)
    (M - (M*densityratio)) % 2 == 0 || throw(error("For the provided arguments, the number of states not in the windowed region is $(M - (M*densityratio)). This value must be even for a symmetric discretisation.")) # constraint enforced such that the a densityratio=0.5 can be utilized and return an integer number of states.
    abs(bandmin) > abs(windmin) || throw(error("Requested window minimum energy lies outside of energy range."))
    abs(bandmax) > abs(windmax) || throw(error("Requested window maximum energy lies outside of energy range."))
    
    M_window = floor(Int, M*densityratio)
    M_sparse = floor(Int, (M - (M*densityratio))/2)

    if fermilevelstate == false
        nothing
    elseif fermilevelstate == true
        M_window = M_window+1 # increase number of states in window region by 1 to place a state at the fermi level
        M_new = M_window + M_sparse * 2
        @info "Additional state added at Fermi level, number of states in discretisation =$M_new"
    else
        throw(error("Invalid value provided to `fermilevelstate`. Only Boolean values allowed."))
    end

    ShenviGauss = ShenviGaussLegendre(M_sparse*2, bandmin - windmin, bandmax - windmax)
    sparse_region = ShenviGauss.bathstates[M_sparse+1:end] .+ windmax

    Trapezoidal = TrapezoidalRule(M_window, windmin, windmax)
    window_region = collect(Trapezoidal.bathstates)

    bathstates = [-1*reverse(sparse_region); window_region; sparse_region]
    bathcoupling = [ShenviGauss.bathcoupling[1:M_sparse]; fill!(copy(window_region), Trapezoidal.bathcoupling); ShenviGauss.bathcoupling[M_sparse+1:end]] 

    return WindowedTrapezoidalRule(bathstates, bathcoupling)
end

"""
    WindowedTrapezoidalRule8(M, bandmin, bandmax, windmin, windmax; densityratio=0.50)

Adaptation of `WindowedTrapezoidalRule8(...)` with a smoother join between the window region and sparse region.                 
"""
function WindowedTrapezoidalRule8(M, bandmin, bandmax, windmin, windmax; densityratio=0.50, couplingcorrection=false)
    (M - (M*densityratio)) % 2 == 0 || throw(error("For the provided arguments, the number of states not in the windowed region is $(M - (M*densityratio)). This value must be even for a symmetric discretisation.")) # constraint enforced such that the a densityratio=0.5 can be utilized and return an integer number of states.
    abs(bandmin) > abs(windmin) || throw(error("Requested window minimum energy lies outside of energy range."))
    abs(bandmax) > abs(windmax) || throw(error("Requested window maximum energy lies outside of energy range."))
    
    M_window = floor(Int, M*densityratio)
    M_sparse = floor(Int, (M - (M*densityratio))/2)

    # densely distributed trapezoidal rule discretised bath states within energy window
    ΔE_window = windmax - windmin
    bstates_window = collect(range(windmin, windmax, length=M_window))

    # window spacing
    δE_window = bstates_window[2] - bstates_window[1]

    # sparse region energy spacing 
    δE_sparse = 2*(bandmax - windmax - 0.5*δE_window) / (2*M_sparse -1)
    
    # sparse to window energy spacing
    δE_join = 0.5*(δE_sparse + δE_window)

    # Sparsely distributed trapezoidal rule discretised bath states
    ΔE_sparse1 = windmin - δE_join  - bandmin # Energy range for first sparsely distributed state region
    bstates_sparse1 = collect(range(bandmin, windmin - δE_join, length=M_sparse)) 

    ΔE_sparse2 = bandmax - windmax - δE_join # Energy Range for second sparsely distributed state region
    bstates_sparse2 = collect(range(windmax + δE_join, bandmax, length=M_sparse))
    

    bathstates = [bstates_sparse1; bstates_window; bstates_sparse2] # concatenates arrays vertically (along axis = 1)    
    
    if couplingcorrection == true
        MiddleIndex = floor(Int, M/2)
        δE = bathstates[MiddleIndex+2:end] .- bathstates[MiddleIndex+1:end-1]
        δE = [δE[1]; δE]
        lb = bathstates[MiddleIndex+1:end] .- (δE .* 0.5)
        ub = bathstates[MiddleIndex+1:end] .+ (δE .* 0.5)
        coupling_sq = (0.5 ./ bathstates[MiddleIndex+1:end]) .* (ub.^2 .- lb.^2) 
        coupling = sqrt.(coupling_sq)
        bathcoupling = [reverse(copy(coupling)); coupling]
    elseif couplingcorrection == false
        bcoupling_window = fill!(copy(bstates_window), sqrt(ΔE_window / (M_window-1)))
        bcoupling_sparse1 = fill!(copy(bstates_sparse1), sqrt(ΔE_sparse1 / (M_sparse-1)))
        bcoupling_sparse2 = fill!(copy(bstates_sparse2), sqrt(ΔE_sparse2 / (M_sparse-1)))
        bathcoupling = [bcoupling_sparse1; bcoupling_window; bcoupling_sparse2]

    else
        throw(error("Invalid value provided to `couplingcorrection`. Only Boolean values allowed."))
    end

    return WindowedTrapezoidalRule(bathstates, bathcoupling)
end



# ------------------------------------------------------------------------------------------------ #

"""
    ShenviGaussLegendre{T}

Defined as described by Shenvi et al. in J. Chem. Phys. 130, 174107 (2009).
The position of the negative sign for the state energy level has been moved to ensure the states are sorted from lowest to highest.
"""
struct ShenviGaussLegendre{T} <: WideBandBathDiscretisation
    bathstates::Vector{T}
    bathcoupling::Vector{T}
end

function ShenviGaussLegendre(M, bandmin, bandmax)
    M % 2 == 0 || throw(error("The number of states `M` must be even."))
    knots, weights = gausslegendre(div(M, 2))
    centre = (bandmax + bandmin) / 2

    bathstates = zeros(M)
    for i in eachindex(knots)
        bathstates[i] = (centre - bandmin)/2 * knots[i] + (bandmin + centre) / 2
    end
    for i in eachindex(knots)
        bathstates[i+length(knots)] = (bandmax - centre)/2 * knots[i] + (bandmax + centre) / 2
    end

    bathcoupling = zeros(M)
    for (i, w) in enumerate(weights)
        bathcoupling[i] = sqrt((centre - bandmin)/2  * w)
    end
    for (i, w) in enumerate(weights)
        bathcoupling[i+length(weights)] = sqrt((bandmax - centre)/2  * w)
    end

    return ShenviGaussLegendre(bathstates, bathcoupling)
end

"""
    ReferenceGaussLegendre{T}

Implementation translated from Fortran code used for simulations of Shenvi et al. in J. Chem. Phys. 130, 174107 (2009).
Two differences from ShenviGaussLegendre:
- Position of minus sign in energy levels has been corrected.
- Division by sqrt(ΔE) in the coupling. 
"""
struct ReferenceGaussLegendre{T} <: WideBandBathDiscretisation
    bathstates::Vector{T}
    bathcoupling::Vector{T}
end

function ReferenceGaussLegendre(M, bandmin, bandmax)
    M % 2 == 0 || throw(error("The number of states `M` must be even."))
    @warn "(ReferenceGaussLegendre) The division by sqrt(ΔE) in the coupling is incorrect. Use ShenviGaussLegendre instead."
    knots, weights = gausslegendre(div(M, 2))
    ΔE = bandmax - bandmin

    bathstates = zeros(M)
    for i in eachindex(knots)
        bathstates[i] = ΔE/2 * (-1/2 + knots[i]/2)
    end
    for i in eachindex(knots)
        bathstates[i+length(knots)] = ΔE/2 * (1/2 + knots[i]/2)
    end

    bathcoupling = zeros(M)
    for (i, w) in zip(eachindex(bathcoupling), Iterators.cycle(weights))
        bathcoupling[i] = sqrt(w) / 2
    end

    return ReferenceGaussLegendre(bathstates, bathcoupling)
end

"""
    FullGaussLegendre{T} <: WideBandBathDiscretisation

Use Gauss-Legendre quadrature to discretise the continuum across the entire band width.
This is similar to the ShenviGaussLegendre except that splits the continuum at the Fermi level into two halves.
"""
struct FullGaussLegendre{T} <: WideBandBathDiscretisation
    bathstates::Vector{T}
    bathcoupling::Vector{T}
end

function FullGaussLegendre(M, bandmin, bandmax)

    knots, weights = gausslegendre(M)
    ΔE = bandmax - bandmin

    bathstates = zeros(M)
    for i in eachindex(knots, bathstates)
        bathstates[i] = ΔE/2 * knots[i]
    end

    bathcoupling = zeros(M)
    for i in eachindex(bathcoupling, weights)
        bathcoupling[i] = sqrt(weights[i] * ΔE/2)
    end

    return FullGaussLegendre(bathstates, bathcoupling)
end

"""
The **`GapGaussLegendre`** struct, a subtype of `WideBandBathDiscretisation`, models a wide-band limit bath with a central energy gap using Gauss–Legendre quadrature for discretization.

## Fields

- **`bathstates::Vector{T}`**: The discretized energy levels of the bath.
- **`bathcoupling::Vector{T}`**: The coupling strengths to the bath states, derived from Gauss–Legendre weights.

## Constructor

```julia
GapGaussLegendre(M, bandmin, bandmax, gapwidth)
```

This constructor validates input arguments and initializes a `GapGaussLegendre` object using Gauss–Legendre quadrature for state placement and coupling computation.

## Arguments

- `M::Int`: The total number of bath states; must be even.
- `bandmin::Real`: The minimum energy of the wide band.
- `bandmax::Real`: The maximum energy of the wide band.
- `gapwidth::Real`: The width of the central energy gap. Must be non-negative and smaller than `bandmax - bandmin`.

## Details

The energy band is split into two symmetric regions around the gap, and `M/2` states are placed in each region using Gauss–Legendre nodes mapped to the corresponding subinterval. The gap itself is avoided.

The `bathcoupling` vector is computed from the square roots of the Gauss–Legendre weights, scaled by the effective half-bandwidth. This ensures accurate integration over the energy band while excluding the gap.
"""
struct GapGaussLegendre{T} <: WideBandBathDiscretisation
    bathstates::Vector{T}
    bathcoupling::Vector{T}
end

function GapGaussLegendre(M, bandmin, bandmax, gapwidth)
    gapwidth < bandmax - bandmin || throw(error("The gap width must be smaller than the band width."))
    gapwidth >= 0 || throw(error("The gap width must be positive."))
    M % 2 == 0 || throw(error("The number of states `M` must be even."))
    knots, weights = gausslegendre(div(M, 2))
    centre = (bandmax + bandmin) / 2
    ΔE = bandmax - bandmin
    bandlengths = (ΔE - gapwidth)/2

    bathstates = zeros(M)
    for i in eachindex(knots)
        bathstates[length(knots)+1-i] = ((ΔE-gapwidth)/2 * (1 + knots[i]) + gapwidth) * (-1/2) + centre
    end
    for i in eachindex(knots)
        bathstates[i+length(knots)] = ((ΔE-gapwidth)/2 * (1 + knots[i]) + gapwidth) * 1/2 + centre
    end

    # coupling is the sqrt() of the weights times the bandlengths [valence band's coupling ; conduction band's coupling]
    bathcoupling = [sqrt.(weights.*bandlengths/2); sqrt.(weights.*bandlengths/2)]

    return GapGaussLegendre(bathstates, bathcoupling)
end

"""
The **`GapTrapezoidalRule`** struct, a subtype of `WideBandBathDiscretisation`, models a wide-band limit bath with an energy gap.

## Fields

- **`bathstates::Vector{T}`**: The discretized energy levels of the bath.
- **`bathcoupling::T`**: The coupling strength between the system and the bath states.

## Constructor

```julia
GapTrapezoidalRule(M, bandmin, bandmax, gapwidth)
```
This constructor validates inputs and initializes a GapTrapezoidalRule object.

## Arguments

- `M::Int`: The total number of bath states; must be even.
- `bandmin::Real`: The minimum energy of the wide band.
- `bandmax::Real`: The maximum energy of the wide band.
- `gapwidth::Real`: The width of the central energy gap. Must be non-negative and smaller than `bandmax - bandmin`.

## Details

If `gapwidth` is zero, `bathstates` are uniformly distributed across the entire band. Otherwise, the band is split, and `M` states are evenly distributed across the two resulting halves, avoiding the gap.

The `bathcoupling` is calculated from the effective bandwidth (`bandmax - bandmin - gapwidth`) and `M`.

"""


struct GapTrapezoidalRule{T} <: WideBandBathDiscretisation
    bathstates::Vector{T}   # ϵ
    bathcoupling::T # V(ϵ,x̃) 
end

function GapTrapezoidalRule(M, bandmin, bandmax, gapwidth)
    gapwidth < bandmax - bandmin || throw(error("The gap width must be smaller than the band width."))
    gapwidth >= 0 || throw(error("The gap width must be positive."))
    M % 2 == 0 || throw(error("The number of states `M` must be even."))
    ΔE = bandmax - bandmin
    centre = (bandmax + bandmin) / 2

    if gapwidth == 0
        bathstates = collect(range(bandmin, bandmax, length=M))
    else
        bathstates_left = range(bandmin, centre - gapwidth/2, length=Int(M/2))
        bathstates_right = range(centre + gapwidth/2, bandmax, length=Int(M/2))
        bathstates = vcat(bathstates_left, bathstates_right)
    end

    bathcoupling = sqrt((ΔE-gapwidth) / M)
    return GapTrapezoidalRule(bathstates, bathcoupling)
end
