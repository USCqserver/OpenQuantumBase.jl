"""
$(TYPEDEF)

Defines a time dependent Hamiltonian object using Julia arrays.

# Fields

$(FIELDS)
"""
struct DenseHamiltonian{T<:Number,dimensionless_time} <: AbstractDenseHamiltonian{T}
    "List of time dependent functions"
    f::Vector
    "List of constant matrices"
    m::Vector
    "Internal cache"
    u_cache::Matrix{T}
    "Size"
    size::Tuple
end

"""
$(SIGNATURES)

Constructor of the `DenseHamiltonian` type. `funcs` and `mats` are lists of time-dependent functions and the corresponding matrices. The Hamiltonian can be represented as ``∑ᵢfuncs[i](s)×mats[i]``. 

`unit` specifies wether `:h` or `:ħ` is set to one when defining `funcs` and `mats`. The `mats` will be scaled by ``2π`` if unit is `:h`.

`dimensionless_time` specifies wether the arguments of the functions are dimensionless (normalized to total evolution time).
"""
function DenseHamiltonian(funcs, mats; unit=:h, dimensionless_time=true)
    if any((x) -> size(x) != size(mats[1]), mats)
        throw(ArgumentError("Matrices in the list do not have the same size."))
    end
    if is_complex(funcs, mats)
        mats = complex.(mats)
    end
    hsize = size(mats[1])
    mats = unit_scale(unit) * mats
    cache = similar(mats[1])
    DenseHamiltonian{eltype(mats[1]),dimensionless_time}(funcs, mats, cache, hsize)
end

"""
    function (h::DenseHamiltonian)(s::Real)

Calling the Hamiltonian returns the value ``2πH(s)``. The argument `s` is the (dimensionless) time. The returned matrix is in the unit of angular frequency.
"""
function (h::DenseHamiltonian)(s::Real)
    fill!(h.u_cache, 0.0)
    for i = 1:length(h.f)
        @inbounds axpy!(h.f[i](s), h.m[i], h.u_cache)
    end
    h.u_cache
end

# The third argument is not essential for `DenseHamiltonian`
# It exists to keep the `update_cache!` interface consistent across
# all `AbstractHamiltonian` types
function update_cache!(cache, H::DenseHamiltonian, ::Any, s::Real)
    fill!(cache, 0.0)
    for i = 1:length(H.m)
        @inbounds axpy!(-1.0im * H.f[i](s), H.m[i], cache)
    end
end

function update_vectorized_cache!(cache, H::DenseHamiltonian, ::Any, s::Real)
    hmat = H(s)
    iden = one(hmat)
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end

function (h::DenseHamiltonian)(du, u::AbstractMatrix, ::Any, s::Real)
    fill!(du, 0.0 + 0.0im)
    H = h(s)
    gemm!('N', 'N', -1.0im, H, u, 1.0 + 0.0im, du)
    gemm!('N', 'N', 1.0im, u, H, 1.0 + 0.0im, du)
end

function Base.convert(S::Type{T}, H::DenseHamiltonian{M}) where {T<:Complex,M}
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, complex{M})
    DenseHamiltonian{eltype(mats[1]),isdimensionlesstime(H)}(H.f, mats, cache, size(H))
end

function Base.convert(S::Type{T}, H::DenseHamiltonian{M}) where {T<:Real,M}
    f_val = sum((x) -> x(0.0), H.f)
    if !(typeof(f_val) <: Real)
        throw(TypeError(:convert, "H.f", Real, typeof(f_val)))
    end
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, real(M))
    DenseHamiltonian{eltype(mats[1]),isdimensionlesstime(H)}(H.f, mats, cache, size(H))
end

function Base.copy(H::DenseHamiltonian)
    mats = copy(H.m)
    DenseHamiltonian{eltype(mats[1])}(H.f, mats, copy(H.u_cache), size(H))
end

function rotate(H::DenseHamiltonian, v)
    mats = [v' * m * v for m in H.m]
    DenseHamiltonian(H.f, mats, unit=:ħ)
end

"""
isdimensionlesstime(H)

Check whether the argument of a time dependent Hamiltonian is the dimensionless time.
"""
isdimensionlesstime(::DenseHamiltonian{T,true}) where {T} = true
isdimensionlesstime(::DenseHamiltonian{T,false}) where {T} = false

"""
$(TYPEDEF)

Defines a time independent Hamiltonian object with a Julia array.

# Fields

$(FIELDS)
"""
struct ConstantDenseHamiltonian{T<:Number} <: AbstractDenseHamiltonian{T}
    "Internal cache"
    u_cache::Matrix{T}
    "Size"
    size::Tuple
end

function ConstantDenseHamiltonian(mat; unit=:h)
    mat = unit_scale(unit) * mat
    ConstantDenseHamiltonian{eltype(mat)}(mat, size(mat))
end

isconstant(::ConstantDenseHamiltonian) = true

function (h::ConstantDenseHamiltonian)(::Real)
    h.u_cache
end

function update_cache!(cache, H::ConstantDenseHamiltonian, ::Any, ::Real)
    cache .= -1.0im * H.u_cache
end

function update_vectorized_cache!(cache, H::ConstantDenseHamiltonian, p, ::Real)
    hmat = H.u_cache
    iden = one(hmat)
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end

function (h::ConstantDenseHamiltonian)(du, u::AbstractMatrix, ::Any, ::Real)
    fill!(du, 0.0 + 0.0im)
    H = h.u_cache
    gemm!('N', 'N', -1.0im, H, u, 1.0 + 0.0im, du)
    gemm!('N', 'N', 1.0im, u, H, 1.0 + 0.0im, du)
end

function Base.convert(S::Type{T}, H::ConstantDenseHamiltonian{M}) where {T<:Number,M}
    mat = convert.(S, H.u_cache)
    ConstantDenseHamiltonian{eltype(mat)}(mat, size(H))
end

function Base.copy(H::ConstantDenseHamiltonian)
    mat = copy(H.u_cache)
    ConstantDenseHamiltonian{eltype(mat[1])}(mat, size(H))
end

function rotate(H::ConstantDenseHamiltonian, v)
    mat = v' * H.u_cache * v
    ConstantDenseHamiltonian(mat, size(mat))
end