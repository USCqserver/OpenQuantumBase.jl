"""
$(TYPEDEF)

Defines a time dependent Hamiltonian object with dense Matrices.

# Fields

$(FIELDS)
"""
struct DenseHamiltonian{T<:Number} <: AbstractDenseHamiltonian{T}
    " List of time dependent functions "
    f::Vector
    " List of constant matrices "
    m::Vector
    """Internal cache"""
    u_cache::AbstractMatrix
    """Size"""
    size::Tuple
    """Eigen decomposition routine"""
    EIGS::Any
end


"""
$(SIGNATURES)

Constructor of DenseHamiltonian object. `funcs` and `mats` are a list of time dependent functions and the corresponding matrices. `unit` is the unit one -- `:h` or `:ħ`. The `mats` will be scaled by ``2π`` if unit is `:h`.

`EIGS` is the initializer for the eigen decomposition routine for the Hamiltonian. It should return a function of signature: `(H, s, lvl) -> (w, v)`. The default method `EIGEN_DEFAULT` will use `LAPACK` routine.
"""
function DenseHamiltonian(funcs, mats; unit = :h, EIGS = EIGEN_DEFAULT)
    if any((x) -> size(x) != size(mats[1]), mats)
        throw(ArgumentError("Matrices in the list do not have the same size."))
    end
    if is_complex(funcs, mats)
        mats = complex.(mats)
    end
    hsize = size(mats[1])
    # use static array for size smaller than 100
    if hsize[1] <= 10
        mats = [SMatrix{hsize[1],hsize[2]}(unit_scale(unit) * m) for m in mats]
    else
        mats = unit_scale(unit) * mats
    end
    cache = similar(mats[1])
    EIGS = EIGS(cache)
    DenseHamiltonian{eltype(mats[1])}(funcs, mats, cache, hsize, EIGS)
end


"""
    function (h::DenseHamiltonian)(s::Real)

Calling the Hamiltonian returns the value ``2πH(s)``. The argument `s` is in the unitless time. The returned matrix is in angular frequency.
"""
function (h::DenseHamiltonian)(s::Real)
    fill!(h.u_cache, 0.0)
    for i = 1:length(h.f)
        @inbounds axpy!(h.f[i](s), h.m[i], h.u_cache)
    end
    h.u_cache
end

# update_func interface for DiffEqOperators
function update_cache!(cache, H::DenseHamiltonian, p, s::Real)
    fill!(cache, 0.0)
    for i = 1:length(H.m)
        @inbounds axpy!(-1.0im * H.f[i](s), H.m[i], cache)
    end
end

function update_vectorized_cache!(cache, H::DenseHamiltonian, p, s::Real)
    hmat = H(s)
    iden = one(hmat)
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end

function (h::DenseHamiltonian)(du, u::AbstractMatrix, p, s::Real)
    fill!(du, 0.0 + 0.0im)
    H = h(s)
    gemm!('N', 'N', -1.0im, H, u, 1.0 + 0.0im, du)
    gemm!('N', 'N', 1.0im, u, H, 1.0 + 0.0im, du)
end

function Base.convert(S::Type{T}, H::DenseHamiltonian) where {T<:Complex}
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, S)
    DenseHamiltonian{eltype(mats[1])}(H.f, mats, cache, size(H), H.EIGS)
end

function Base.convert(S::Type{T}, H::DenseHamiltonian) where {T<:Real}
    f_val = sum((x) -> x(0.0), H.f)
    if !(typeof(f_val) <: Real)
        throw(TypeError(:convert, "H.f", Real, typeof(f_val)))
    end
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, S)
    DenseHamiltonian{eltype(mats[1])}(H.f, mats, cache, size(H), H.EIGS)
end
