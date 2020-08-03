"""
$(TYPEDEF)

Defines constant system bath coupling operators.

# Fields

$(FIELDS)
"""
struct ConstantCouplings <: AbstractCouplings
    """1-D array for independent coupling operators"""
    mats::Vector{AbstractMatrix}
    """String representation for the coupling (for display purpose)"""
    str_rep::Union{Vector{String},Nothing}
end

(c::ConstantCouplings)(t) = c.mats
Base.iterate(c::ConstantCouplings, state = 1) =
    state > length(c.mats) ? nothing : ((x) -> c.mats[state], state + 1)
Base.length(c::ConstantCouplings) = length(c.mats)
Base.eltype(c::ConstantCouplings) = typeof(c.mats[1])
Base.size(c::ConstantCouplings) = size(c.mats[1])
Base.size(c::ConstantCouplings, d) = size(c.mats[1], d)

"""
    function ConstantCouplings(mats; str_rep=nothing, unit=:h)

Constructor of `ConstantCouplings` object. `mats` is 1-D array of matrices. `str_rep` is the optional string representation of the coupling terms. `unit` is the unit one -- `:h` or `:ħ`. The `mats` will be scaled by ``2π`` is unit is `:h`.
"""
function ConstantCouplings(
    mats::Union{Vector{Matrix{T}},Vector{SparseMatrixCSC{T,Int}}};
    unit = :h,
) where {T<:Number}
    msize = size(mats[1])
    if msize[1] <= 10
        if issparse(mats[1])
            @warn "For matrices smaller than 10×10, use StaticArrays by default."
            mats = Array.(mats)
        end
        mats = [SMatrix{msize[1],msize[2]}(unit_scale(unit) * m) for m in mats]
    else
        mats = unit_scale(unit) .* mats
    end
    ConstantCouplings(mats, nothing)
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
    mats = q_translate.(c, sp = sp)
    msize = size(mats[1])
    if msize[1] <= 10
        if sp
            @warn "For matrices smaller than 10×10, use StaticArrays by default."
            mats = Array.(mats)
        end
        mats = [SMatrix{msize[1],msize[2]}(unit_scale(unit) * m) for m in mats]
    else
        mats = unit_scale(unit) .* mats
    end
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
Base.size(c::TimeDependentCoupling, d) = size(c.mats[1], d)

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
Base.size(c::TimeDependentCouplings, d) = size(c.coupling[1], d)

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

function CustomCouplings(funcs; unit = :h)
    mat = funcs[1](0.0)
    if unit == :h
        funcs = [(s) -> 2π * f(s) for f in funcs]
    elseif unit != :ħ
        throw(ArgumentError("The unit can only be :h or :ħ."))
    end
    CustomCouplings(funcs, size(mat))
end

(c::CustomCouplings)(s) = [x(s) for x in c.coupling]
Base.size(c::CustomCouplings) = c.size
Base.size(c::CustomCouplings, d) = c.size[d]
Base.iterate(c::AbstractTimeDependentCouplings, state = 1) =
    Base.iterate(c.coupling, state)
Base.length(c::AbstractTimeDependentCouplings) = length(c.coupling)
Base.eltype(c::AbstractTimeDependentCouplings) = typeof(c.coupling[1])
