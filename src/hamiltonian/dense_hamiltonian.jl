"""
$(TYPEDEF)

Defines a time dependent Hamiltonian object with dense matrices.

# Fields

$(FIELDS)
"""
struct DenseHamiltonian{T<:Number} <: AbstractDenseHamiltonian{T}
    "List of time dependent functions"
    f::Vector
    "List of constant matrices"
    m::Vector
    "Internal cache"
    u_cache::AbstractMatrix{T}
    "Size"
    size::Tuple
end

"""
$(SIGNATURES)

Constructor of the `DenseHamiltonian` type. `funcs` and `mats` are lists of time-dependent functions and the corresponding matrices. The Hamiltonian can be represented as ``∑ᵢfuncs[i](s)×mats[i]``. `unit` specifies wether `:h` or `:ħ` is set to one when defining `funcs` and `mats`. The `mats` will be scaled by ``2π`` if unit is `:h`.
"""
function DenseHamiltonian(funcs, mats; unit = :h)
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
    DenseHamiltonian{eltype(mats[1])}(funcs, mats, cache, hsize)
end

"""
    function (h::DenseHamiltonian)(s::Real)

Calling the Hamiltonian returns the value ``2πH(s)``. The argument `s` is in the dimensionless time. The returned matrix is in the unit of angular frequency.
"""
function (h::DenseHamiltonian)(s::Real)
    fill!(h.u_cache, 0.0)
    for i = 1:length(h.f)
        @inbounds axpy!(h.f[i](s), h.m[i], h.u_cache)
    end
    h.u_cache
end

# The argument `p` is not essential for `DenseHamiltonian`
# It exists to keep the `update_cache!` interface consistent across
# all `AbstractHamiltonian` types
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

function Base.convert(S::Type{T}, H::DenseHamiltonian{M}) where {T<:Complex,M}
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, complex{M})
    DenseHamiltonian{eltype(mats[1])}(H.f, mats, cache, size(H))
end

function Base.convert(S::Type{T}, H::DenseHamiltonian{M}) where {T<:Real,M}
    f_val = sum((x) -> x(0.0), H.f)
    if !(typeof(f_val) <: Real)
        throw(TypeError(:convert, "H.f", Real, typeof(f_val)))
    end
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, real(M))
    DenseHamiltonian{eltype(mats[1])}(H.f, mats, cache, size(H))
end

function Base.copy(H::DenseHamiltonian)
    mats = Base.copy(H.m)
    DenseHamiltonian{eltype(mats[1])}(H.f, mats, Base.copy(H.u_cache), size(H))
end

function rotate(H::DenseHamiltonian, v)
    mats = [v' * m * v for m in H.m]
    DenseHamiltonian(H.f, mats, unit=:ħ)
end

"""
$(TYPEDEF)

Defines a time independent Hamiltonian object with dense matrices.

# Fields

$(FIELDS)
"""
struct ConstantDenseHamiltonian{T<:Number} <: AbstractDenseHamiltonian{T}
    "Internal cache"
    u_cache::AbstractMatrix{T}
    "Size"
    size::Tuple
end

function (h::ConstantDenseHamiltonian)(::Real)
    h.u_cache
end

function update_cache!(cache, H::ConstantDenseHamiltonian, p, ::Real)
    fill!(cache, 0.0)
    axpy!(-1.0im, H.u_cache, cache)
end

function update_vectorized_cache!(cache, H::ConstantDenseHamiltonian, p, ::Real)
    hmat = H.u_cache
    iden = one(hmat)
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end

function (h::ConstantDenseHamiltonian)(du, u::AbstractMatrix, p, s::Real)
    fill!(du, 0.0 + 0.0im)
    H = h.u_cache
    gemm!('N', 'N', -1.0im, H, u, 1.0 + 0.0im, du)
    gemm!('N', 'N', 1.0im, u, H, 1.0 + 0.0im, du)
end

function Base.convert(S::Type{T}, H::ConstantDenseHamiltonian{M}) where {T,M}
    mat = convert.(S, H.u_cache)
    ConstantDenseHamiltonian{eltype(mat)}(mat, size(H))
end

function Base.copy(H::ConstantDenseHamiltonian)
    mat = copy(H.u_cache)
    ConstantDenseHamiltonian{eltype(mat[1])}(mat, size(H))
end

function rotate(H::ConstantDenseHamiltonian, v)
    mat = v' * H.u_cache * v
    DenseHamiltonian(mat, size(mat))
end