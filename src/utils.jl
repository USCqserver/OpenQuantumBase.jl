macro CSI_str(str)
    return :(string("\x1b[", $(esc(str)), "m"))
end


const TYPE_COLOR = CSI"36"
const NO_COLOR = CSI"0"


function is_sparse(H::AbstractHamiltonian)
    typeof(H) <: AbstractSparseHamiltonian
end


"""
$(TYPEDEF)

A internal object to inform the solver to use the physical time instead of the unitless time.

# Fields
$(FIELDS)
"""
struct UnitTime{T<:AbstractFloat}
    """Time in physical unit"""
    t::T
end


UnitTime(x::Real) = UnitTime(float(x))


for op in (:*, :/, :\)
    @eval Base.$op(t::UnitTime, x) = $op(t.t, x)
    @eval Base.$op(x, t::UnitTime) = $op(x, t.t)
end


"""
    function unit_scale(u)

Determine the value of ``1/ħ``. Return ``2π`` if `u` is `:h`; and return 1 if `u` is `:ħ`.
"""
function unit_scale(u)
    if u == :h
        return 2π
    elseif u == :ħ
        return 1
    else
        throw(ArgumentError("The unit can only be :h or :ħ."))
    end
end


"""
    function allequal(x; rtol = 1e-6, atol = 1e-6)

Check if all elements in `x` are equal upto an absolute error tolerance `atol` and relative error tolerance `rtol`.
"""
@inline function allequal(x; rtol = 1e-6, atol = 1e-6)
    length(x) < 2 && return true
    e1 = x[1]
    i = 2
    @inbounds for i = 2:length(x)
        isapprox(x[i], e1, rtol = rtol, atol = atol) || return false
    end
    return true
end
