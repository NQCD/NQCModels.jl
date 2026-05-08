using Unitful, UnitfulAtomic

function lorentzian(energy::Float64, width::Float64)
    return width^2 / (width^2 + energy^2)
end

struct lorentzianbath{T} <: BathFunction
    bathfunction :: Vector{T}
    bathtype :: Symbol
end

function lorentzianbath(discretisation::BathDiscretisationScheme; W=4.5)
    W_au = austrip(W*u"eV")
    bathfunction = lorentzian.(discretisation.bathstates, W_au)
    bathtype = :lorentzian
    return lorentzianbath(bathfunction, bathtype)
end