
abstract type TullyModel <: DiabaticModel end

NonadiabaticModels.ndofs(::TullyModel) = 1
NonadiabaticModels.nstates(::TullyModel) = 2

"""
    TullyModelOne(a=0.01, b=1.6, c=0.005, d=1.0)

Tully's simple avoided crossing model from [J. Chem. Phys. 93, 1061 (1990)](https://doi.org/10.1063/1.459170).
"""
Parameters.@with_kw struct TullyModelOne{A,B,C,D} <: TullyModel
    a::A = 0.01
    b::B = 1.6
    c::C = 0.005
    d::D = 1.0
end

function NonadiabaticModels.potential(model::TullyModelOne, q::Real)
    Parameters.@unpack a, b, c, d = model
    if q > 0
        V11 = a * (1 - exp(-b*q))
    else
        V11 = -a * (1 - exp(b*q))
    end
    V22 = -V11
    V12 = c * exp(-d*q^2) # sets both off-diagonals as V is Hermitian
    return Hermitian(SMatrix{2,2}(V11, V12, V12, V22))
end

function NonadiabaticModels.derivative(model::TullyModelOne, q::Real)
    Parameters.@unpack a, b, c, d = model
    D11 = a * b * exp(-b * abs(q))
    D22 = -D11
    D12 = -2 * c * d * q * exp(-d*q^2)
    return Hermitian(SMatrix{2,2}(D11, D12, D12, D22))
end

"""
    TullyModelTwo(a=0.1, b=0.28, c=0.015, d=0.06, e=0.05)

Tully's dual avoided crossing model from [J. Chem. Phys. 93, 1061 (1990)](https://doi.org/10.1063/1.459170).
"""
Parameters.@with_kw struct TullyModelTwo{A,B,C,D,E} <: TullyModel
    a::A = 0.1
    b::B = 0.28
    c::C = 0.015
    d::D = 0.06
    e::E = 0.05
end

function NonadiabaticModels.potential(model::TullyModelTwo, q::Real)
    Parameters.@unpack a, b, c, d, e = model
    V11 = 0
    V22 = -a*exp(-b*q^2) + e
    V12 = c * exp(-d*q^2)
    return Hermitian(SMatrix{2,2}(V11, V12, V12, V22))
end

function NonadiabaticModels.derivative(model::TullyModelTwo, q::Real)
    Parameters.@unpack a, b, c, d  = model
    D11 = 0
    D22 = 2*a*b*q*exp(-b*q^2)
    D12 = -2*c*d*q*exp(-d*q^2)

    Hermitian(SMatrix{2,2}(D11, D12, D12, D22))
end

"""
    TullyModelThree(a=0.0006, b=0.1, c=0.9)

Tully's extended coupling with reflection model from [J. Chem. Phys. 93, 1061 (1990)](https://doi.org/10.1063/1.459170).
"""
Parameters.@with_kw struct TullyModelThree{A,B,C} <: TullyModel
    a::A = 0.0006
    b::B = 0.1
    c::C = 0.9
end

function NonadiabaticModels.potential(model::TullyModelThree, q::Real)
    Parameters.@unpack a, b, c = model
    V11 = a
    V22 = -a
    if q > 0
        V12 = b * (2 - exp(-c*q))
    else
        V12 = b * exp(c*q)
    end
    return Hermitian(SMatrix{2,2}(V11, V12, V12, V22))
end

function NonadiabaticModels.derivative(model::TullyModelThree, R::Real)
    Parameters.@unpack a, b, c = model
    q = R[1]
    D11 = 0
    D22 = 0
    if q > 0
        D12 = b * c * exp(-c*q)
    else
        D12 = b * c * exp(c*q)
    end
    return Hermitian(SMatrix{2,2}(D11, D12, D12, D22))
end
