"""
$(TYPEDEF)

Defines constant system bath coupling operators.

# Fields

$(FIELDS)
"""
struct ConstantCouplings <: AbstractCouplings
    """1-D array for independent coupling operators"""
    mats::Any
    """String representation for the coupling (for display purpose)"""
    str_rep::Vector{String}
end

(c::ConstantCouplings)(t) = c.mats
Base.iterate(c::ConstantCouplings, state = 1) = Base.iterate(c.mats, state)
Base.length(c::ConstantCouplings) = length(c.mats)
Base.eltype(c::ConstantCouplings) = typeof(c.mats[1])
Base.size(c::ConstantCouplings) = size(c.mats[1])

"""
    function ConstantCouplings(mats; str_rep=nothing, unit=:h)

Constructor of `ConstantCouplings` object. `mats` is 1-D array of matrices. `str_rep` is the optional string representation of the coupling terms. `unit` is the unit one -- `:h` or `:ħ`. The `mats` will be scaled by ``2π`` is unit is `:h`.
"""
function ConstantCouplings(
    mats::Union{Vector{Matrix{T}},Vector{SparseMatrixCSC{T,Int}}};
    str_rep = nothing,
    unit = :h,
) where {T<:Number}
    if str_rep != nothing
        for s in str_rep
            if !(typeof(s) <: AbstractString)
                throw(ArgumentError("String representation can only be strings."))
            end
        end
    end
    ConstantCouplings(unit_scale(unit) * mats, str_rep)
end

"""
    function ConstantCouplings(c::Vector{T}; sp = false, unit=:h) where T <: AbstractString

If the first argument is a 1-D array of strings. The constructor will automatically construct the matrics represented by the string representations.
"""
function ConstantCouplings(
    c::Vector{T};
    sp = false,
    unit = :h,
) where {T<:AbstractString}
    mats = unit_scale(unit) * q_translate.(c, sp = sp)
    ConstantCouplings(mats, c)
end


"""
    function collective_coupling(op, num_qubit; sp=false)

Create `ConstantCouplings` object with operator `op` on each qubits. `op` is the string representation of one of the Pauli matrices. `num_qubit` is the total number of qubits. `sp` set whether to use sparse matrices. `unit` set the unit one -- ``h`` or ``ħ``.
"""
function collective_coupling(op, num_qubit; sp = false, unit = :h)
    res = Vector{String}()
    for i = 1:num_qubit
        temp = "I"^(i - 1) * uppercase(op) * "I"^(num_qubit - i)
        push!(res, temp)
    end
    ConstantCouplings(res; sp = sp, unit = unit)
end

Base.summary(C::ConstantCouplings) = string(
    TYPE_COLOR,
    nameof(typeof(C)),
    NO_COLOR,
    " with ",
    TYPE_COLOR,
    typeof(C.mats).parameters[1],
    NO_COLOR,
)

function Base.show(io::IO, C::ConstantCouplings)
    println(io, summary(C))
    print(io, "and string representation: ")
    show(io, C.str_rep)
end

"""
$(TYPEDEF)

Defines a single time dependent system bath coupling operator. It is defined as ``S(s)=∑f(s)×M``.  Keyword argument `unit` set the unit one -- ``h`` or ``ħ``.

# Fields

$(FIELDS)
"""
struct TimeDependentCoupling
    """1-D array of time dependent functions"""
    funcs::Any
    """1-D array of constant matrics"""
    mats::Any

    function TimeDependentCoupling(funcs, mats; unit = :h)
        new(funcs, unit_scale(unit) * mats)
    end
end

(c::TimeDependentCoupling)(t) = sum((x) -> x[1](t) * x[2], zip(c.funcs, c.mats))
Base.size(c::TimeDependentCoupling) = size(c.mats[1])

abstract type AbstractTimeDependentCouplings <: AbstractCouplings end

"""
$(TYPEDEF)

Defines an 1-D array of time dependent system bath coupling operators.

# Fields

$(FIELDS)
"""
struct TimeDependentCouplings <: AbstractTimeDependentCouplings
    """A tuple of single `TimeDependentCoupling` operators"""
    coupling::Tuple

    function TimeDependentCouplings(args...)
        new(args)
    end
end

(c::TimeDependentCouplings)(t) = [x(t) for x in c.coupling]
Base.size(c::TimeDependentCouplings) = size(c.coupling[1])

"""
$(TYPEDEF)

`CustomCouplings` is a container for any user defined coupling operators.

# Fields

$(FIELDS)
"""
struct CustomCouplings <: AbstractTimeDependentCouplings
    """A 1-D array of callable objects that returns coupling matrices"""
    coupling::Any
    """Size of the coupling operator"""
    size::Any
end

function CustomCouplings(funcs)
    mat = funcs[1](0.0)
    CustomCouplings(funcs, size(mat))
end

(c::CustomCouplings)(s) = [x(s) for x in c.coupling]
Base.size(c::CustomCouplings) = c.size
Base.iterate(c::AbstractTimeDependentCouplings, state = 1) =
    Base.iterate(c.coupling, state)
Base.length(c::AbstractTimeDependentCouplings) = length(c.coupling)
Base.eltype(c::AbstractTimeDependentCouplings) = typeof(c.coupling[1])
