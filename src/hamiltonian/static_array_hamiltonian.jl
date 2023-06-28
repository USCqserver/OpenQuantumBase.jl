"""
$(TYPEDEF)

Defines a time dependent Hamiltonian object using static arrays.

# Fields

$(FIELDS)
"""
struct StaticDenseHamiltonian{T<:Number,dimensionless_time} <: AbstractDenseHamiltonian{T}
    "List of time dependent functions"
    f::Vector
    "List of constant matrices"
    m::Vector
    "Internal cache"
    u_cache::MMatrix
    "Size"
    size::Tuple
end

function StaticDenseHamiltonian(funcs, mats; unit=:h, dimensionless_time=true)
    if any((x) -> size(x) != size(mats[1]), mats)
        throw(ArgumentError("Matrices in the list do not have the same size."))
    end
    if is_complex(funcs, mats)
        mats = complex.(mats)
    end
    hsize = size(mats[1])
    mats = unit_scale(unit) * mats
    cache = similar(mats[1])
    StaticDenseHamiltonian{eltype(mats[1]),dimensionless_time}(funcs, mats, cache, hsize)
end

isdimensionlesstime(::StaticDenseHamiltonian{T,B}) where {T,B} = B
issparse(::StaticDenseHamiltonian) = false

"""
    function (h::StaticDenseHamiltonian)(s::Real)

Calling the Hamiltonian returns the value ``2πH(s)``. The argument `s` is (dimensionless) time. The returned matrix is in the unit of angular frequency.
"""
function (h::StaticDenseHamiltonian)(s::Real)
    fill!(h.u_cache, 0.0)
    for i = 1:length(h.f)
        @inbounds axpy!(h.f[i](s), h.m[i], h.u_cache)
    end
    h.u_cache
end

# The third argument is not essential for `StaticDenseHamiltonian`
# It exists to keep the `update_cache!` interface consistent across
# all `AbstractHamiltonian` types
function update_cache!(cache, H::StaticDenseHamiltonian, ::Any, s::Real)
    fill!(cache, 0.0)
    for i = 1:length(H.m)
        @inbounds axpy!(-1.0im * H.f[i](s), H.m[i], cache)
    end
end

function update_vectorized_cache!(cache, H::StaticDenseHamiltonian, ::Any, s::Real)
    hmat = H(s)
    iden = one(hmat)
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end

function (h::StaticDenseHamiltonian)(du, u::AbstractMatrix, ::Any, s::Real)
    fill!(du, 0.0 + 0.0im)
    H = h(s)
    gemm!('N', 'N', -1.0im, H, u, 1.0 + 0.0im, du)
    gemm!('N', 'N', 1.0im, u, H, 1.0 + 0.0im, du)
end

function Base.convert(S::Type{T}, H::StaticDenseHamiltonian{M}) where {T<:Complex,M}
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, complex{M})
    StaticDenseHamiltonian{eltype(mats[1])}(H.f, mats, cache, size(H))
end

function Base.convert(S::Type{T}, H::StaticDenseHamiltonian{M}) where {T<:Real,M}
    f_val = sum((x) -> x(0.0), H.f)
    if !(typeof(f_val) <: Real)
        throw(TypeError(:convert, "H.f", Real, typeof(f_val)))
    end
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, real(M))
    StaticDenseHamiltonian{eltype(mats[1])}(H.f, mats, cache, size(H))
end

function Base.copy(H::StaticDenseHamiltonian)
    mats = copy(H.m)
    StaticDenseHamiltonian{eltype(mats[1])}(H.f, mats, copy(H.u_cache), size(H))
end

function rotate(H::StaticDenseHamiltonian, v)
    hsize = size(H)
    mats = [SMatrix{hsize[1],hsize[2]}(v' * m * v) for m in H.m]
    StaticDenseHamiltonian(H.f, mats, unit=:ħ, dimensionless_time=isdimensionlesstime(H))
end

function haml_eigs_default(H::StaticDenseHamiltonian, t, lvl::Integer)
    w, v = eigen(Hermitian(H(t)))
    w[1:lvl], v[:, 1:lvl]
end

"""
$(TYPEDEF)

Defines a time independent Hamiltonian object using static arrays.

# Fields

$(FIELDS)
"""
struct ConstantStaticDenseHamiltonian{T<:Number} <: AbstractDenseHamiltonian{T}
    "Internal cache"
    u_cache::AbstractMatrix{T}
    "Size"
    size::Tuple
end

function ConstantStaticDenseHamiltonian(mat; unit=:h)
    mat = unit_scale(unit) * mat
    ConstantStaticDenseHamiltonian{eltype(mat)}(mat, size(mat))
end

isconstant(::ConstantStaticDenseHamiltonian) = true
issparse(::ConstantStaticDenseHamiltonian) = false

function (h::ConstantStaticDenseHamiltonian)(::Real)
    h.u_cache
end

function update_cache!(cache, H::ConstantStaticDenseHamiltonian, ::Any, ::Real)
    cache .= -1.0im * H.u_cache
end

function update_vectorized_cache!(cache, H::ConstantStaticDenseHamiltonian, p, ::Real)
    hmat = H.u_cache
    iden = one(hmat)
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end

function (h::ConstantStaticDenseHamiltonian)(du, u::AbstractMatrix, ::Any, ::Real)
    fill!(du, 0.0 + 0.0im)
    H = h.u_cache
    du .= -1.0im * (H * u - u * H)
end

function Base.convert(S::Type{T}, H::ConstantStaticDenseHamiltonian{M}) where {T<:Number,M}
    mat = convert.(S, H.u_cache)
    ConstantStaticDenseHamiltonian{eltype(mat)}(mat, size(H))
end

function Base.copy(H::ConstantStaticDenseHamiltonian)
    mat = copy(H.u_cache)
    ConstantStaticDenseHamiltonian{eltype(mat[1])}(mat, size(H))
end

function rotate(H::ConstantStaticDenseHamiltonian, v)
    hsize = size(H)
    mat = SMatrix{hsize[1],hsize[2]}(v' * H.u_cache * v)
    ConstantStaticDenseHamiltonian(mat, hsize)
end

function haml_eigs_default(H::ConstantStaticDenseHamiltonian, t, lvl::Integer)
    w, v = eigen(Hermitian(H(t)))
    w[1:lvl], v[:, 1:lvl]
end