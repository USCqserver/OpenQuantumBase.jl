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
@inline function allequal(x; rtol = 1e-6, atol = 1e-6)
    length(x) < 2 && return true
    e1 = x[1]
    i = 2
    @inbounds for i = 2:length(x)
        isapprox(x[i], e1, rtol = rtol, atol = atol) || return false
    end
    return true
end

"""
    function EIGEN_DEFAULT(H)

The default initializer for eigen factorization method. It returns a function of signature: `(H, s, lvl) -> (w, v)`. `H` is the Hamiltonian object, `s` is the dimensionless time and `lvl` is the energy levels to keep. This default initializer will use `LAPACK` routine for both dense and sparse matrices.
"""
function EIGEN_DEFAULT(H::AbstractDenseHamiltonian)
    function (H, t, lvl)
        hmat = H(t)
        eigen!(Hermitian(hmat), 1:lvl)
    end
end

function EIGEN_DEFAULT(H::AbstractSparseHamiltonian)
    function (H, t, lvl)
        hmat = H(t)
        eigen!(Hermitian(Array(hmat)), 1:lvl)
    end
end

function numargs(f)
    numparam = maximum([num_types_in_tuple(m.sig) for m in methods(f)])
    return (numparam - 1) #-1 in v0.5 since it adds f as the first parameter
end

function num_types_in_tuple(sig)
    length(sig.parameters)
end

function num_types_in_tuple(sig::UnionAll)
    length(Base.unwrap_unionall(sig).parameters)
end
