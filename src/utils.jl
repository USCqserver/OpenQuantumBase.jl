macro CSI_str(str)
    return :(string("\x1b[", $(esc(str)), "m"))
end

const TYPE_COLOR = CSI"36"
const NO_COLOR = CSI"0"

function is_sparse(H::AbstractHamiltonian)
    typeof(H)<:AbstractSparseHamiltonian
end

struct UnitTime{T<:AbstractFloat}
    t::T
end

function UnitTime(x::Real)
    UnitTime(float(x))
end

function Base.:*(a::UnitTime, b)
    a.t * b
end

function Base.:*(b, a::UnitTime)
    b * a.t
end

function Base.:/(b, a::UnitTime)
    b / a.t
end
