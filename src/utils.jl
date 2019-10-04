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
