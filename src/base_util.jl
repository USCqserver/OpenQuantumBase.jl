macro CSI_str(str)
    return :(string("\x1b[", $(esc(str)), "m"))
end

const TYPE_COLOR = CSI"36"
const NO_COLOR = CSI"0"
issparse(H::AbstractHamiltonian) = typeof(H) <: AbstractSparseHamiltonian

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
@inline function allequal(x; rtol=1e-6, atol=1e-6)
    length(x) < 2 && return true
    e1 = x[1]
    i = 2
    @inbounds for i = 2:length(x)
        isapprox(x[i], e1, rtol=rtol, atol=atol) || return false
    end
    return true
end

function numargs(f)
    numparam = maximum([num_types_in_tuple(m.sig) for m in methods(f)])
    return (numparam - 1) # -1 in v0.5 since it adds f as the first parameter
end

function num_types_in_tuple(sig)
    length(sig.parameters)
end

function num_types_in_tuple(sig::UnionAll)
    length(Base.unwrap_unionall(sig).parameters)
end

"""
$(TYPEDEF)

A statistical ensemble that is equivalent to a density matrix ρ.

# Fields

$(FIELDS)
"""
struct EᵨEnsemble
    "The weights for each state vector"
    ω::Vector
    "The states vectors in the ensemble"
    vec::Vector
    function EᵨEnsemble(ω::Vector, vec::Vector)
        if !(eltype(ω) <: Number && all((x) -> x >= 0, ω))
            throw(ArgumentError("Weights vector ω must be 1-d array of positive numbers."))
        end
        if !(eltype(vec) <: Vector)
            throw(ArgumentError("vec must be 1-d array of vectors."))
        end
        new(ω, vec)
    end
end

sample_state_vector(Eᵨ::EᵨEnsemble) = sample(Eᵨ.vec, Weights(Eᵨ.ω))

"""
$(TYPEDEF)

A container for a matrix whose elements are the same function. Getting any indices of this type returns the same function.

$(FIELDS)
"""
struct SingleFunctionMatrix
    "Internal function"
    fun::Any
end
@inline Base.getindex(F::SingleFunctionMatrix, ind...) = F.fun


"""
$(SIGNATURES)

Convert bloch sphere angle `θ` and `ϕ` to the corresponding state vector according to ``cos(θ/2)|0⟩+exp(iϕ)sin(θ/2)|1⟩``.
"""
function bloch_to_state(θ::Real, ϕ::Real)
    if !(0<=θ<=π)
        throw(ArgumentError("θ is out of range 0≤θ≤π."))
    end
    if !(0<=ϕ<=2π)
        throw(ArgumentError("ϕ is out of range 0≤ϕ≤2π."))
    end
    cos(θ / 2) * PauliVec[3][1] + exp(1.0im * ϕ) * sin(θ / 2) * PauliVec[3][2]
end

"""
$(SIGNATURES)

Split a Pauli expression (a string like "-0.1X1X2+Z1") into a list of substrings
that represent each Pauli clause. Each substring contains three parts: the sign;
the prefix; and the Pauli string.

# Example
```julia-repl
julia> split_pauli_expression("-0.1X1X2+Z2")
2-element Vector{Any}:
SubString{String}["-", "0.1", "X1X2"]
SubString{String}["+", "", "Z2"]
```
"""
function split_pauli_expression(p_str)
    p_str = filter((x)->!isspace(x), p_str)
    re_str = r"(\+|-|^)([0-9im.]*|^)([XYZ0-9]+)"
    res = []
    for m in eachmatch(re_str, p_str)
        push!(res, [m[1], m[2], m[3]])
    end
    res
end
