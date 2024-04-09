"""
$(TYPEDEF)

Defines a time dependent Hamiltonian object with sparse matrices.

# Fields

$(FIELDS)
"""
struct SparseHamiltonian{T<:Number,dimensionless_time} <: AbstractHamiltonian{T}
    "List of time dependent functions"
    f::Any
    "List of constant matrices"
    m::Vector{SparseMatrixCSC{T,Int}}
    "Internal cache"
    u_cache::SparseMatrixCSC{T,Int}
    "Size"
    size::Tuple
end

"""
$(SIGNATURES)

Constructor of the `SparseHamiltonian` type. `funcs` and `mats` are lists of time-dependent functions and the corresponding matrices. The Hamiltonian can be represented as ``∑ᵢfuncs[i](s)×mats[i]``. `unit` specifies wether `:h` or `:ħ` is set to one when defining `funcs` and `mats`. The `mats` will be scaled by ``2π`` if unit is `:h`.
"""
function SparseHamiltonian(funcs, mats; unit=:h, dimensionless_time=true)
    if any((x) -> size(x) != size(mats[1]), mats)
        throw(ArgumentError("Matrices in the list do not have the same size."))
    end
    if is_complex(funcs, mats)
        mats = complex.(mats)
    end
    cache = similar(sum(mats))
    fill!(cache, 0.0)
    mats = unit_scale(unit) * mats
    SparseHamiltonian{eltype(mats[1]),dimensionless_time}(funcs, mats, cache, size(mats[1]))
end

function Base.:+(h1::SparseHamiltonian, h2::SparseHamiltonian)
    @assert size(h1) == size(h2) "The two Hamiltonians need to have the same size."
    @assert isdimensionlesstime(h1) == isdimensionlesstime(h2) "The two Hamiltonians need to have the time arguments."

    (m1, m2) = promote(h1.m, h2.m)
    mats = [m1; m2]
    cache = similar(sum(mats))
    funcs = [h1.f; h2.f]
    hsize = size(h1)
    SparseHamiltonian{eltype(m1[1]),isdimensionlesstime(h1)}(funcs, mats, cache, hsize)
end

isdimensionlesstime(::SparseHamiltonian{T,B}) where {T,B} = B
issparse(::SparseHamiltonian) = true

"""
    function (h::SparseHamiltonian)(t::Real)

Calling the Hamiltonian returns the value ``2πH(t)``.
"""
function (h::SparseHamiltonian)(s::Real)
    fill!(h.u_cache, 0.0)
    for (f, m) in zip(h.f, h.m)
        h.u_cache .+= f(s) * m
    end
    h.u_cache
end

# The third argument is not essential for `SparseHamiltonian`
# It exists to keep the `update_cache!` interface consistent across
# all `AbstractHamiltonian` types
function update_cache!(cache, H::SparseHamiltonian, ::Any, s::Real)
    fill!(cache, 0.0)
    for (f, m) in zip(H.f, H.m)
        cache .+= -1.0im * f(s) * m
    end
end

function update_vectorized_cache!(cache, H::SparseHamiltonian, ::Any, s::Real)
    hmat = H(s)
    iden = sparse(I, size(H))
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end

function (h::SparseHamiltonian)(
    du,
    u::AbstractMatrix,
    ::Any,
    s::Real,
)
    H = h(s)
    du .= -1.0im * (H * u - u * H)
end

function Base.convert(S::Type{T}, H::SparseHamiltonian{M}) where {T<:Complex,M}
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, complex{M})
    SparseHamiltonian{eltype(mats[1]),isdimensionlesstime(H)}(H.f, mats, cache, size(H))
end

function Base.convert(S::Type{T}, H::SparseHamiltonian{M}) where {T<:Real,M}
    f_val = sum((x) -> x(0.0), H.f)
    if !(typeof(f_val) <: Real)
        throw(TypeError(:convert, "H.f", Real, typeof(f_val)))
    end
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, real(M))
    SparseHamiltonian{eltype(mats[1]),isdimensionlesstime(H)}(H.f, mats, cache, size(H))
end

"""
    haml_eigs_default(H::SparseHamiltonian, t, lvl::Integer; kwargs...)

Perform the eigendecomposition of a sparse Hamiltonian `H` at a given time `t`.
The argument `lvl` specifies the number of levels to retain in the output. This
method utilizes the LOBPCG algorithm when the size `d` of the Hamiltonian
satisfies `d >= 3 * lvl`, and Julia's built-in `eigen` function otherwise.

The function returns a tuple `(λ, X)`, where `λ` is a vector of eigenvalues and
`X` is a matrix where each column represents an eigenvector.

Keyword arguments `kwargs...` can be used to pass additional parameters to the 
LOBPCG algorithm.
"""
function haml_eigs_default(H::SparseHamiltonian{T,B}, t, lvl::Integer;
    lobpcg=true, kwargs...) where {T<:Number,B}
    d = size(H, 1)
    # LOBPCG algorithm only works when 3*lvl >= d
    # TODO: filter the keyword arguments
    if d >= 3 * lvl && lobpcg
        X0 = randn(T, (d, lvl))
        res = lobpcg_hyper(H(t), X0; kwargs...)
        return (res.λ, res.X)
    else
        return eigen(H(t) |> Array |> Hermitian)
    end
end

"""
    haml_eigs_default(H::SparseHamiltonian, t, X0::Matrix, kwargs...)

Perform the eigendecomposition of a sparse Hamiltonian `H` at a given time `t` 
using an initial guess `X0` for the eigenvectors. This method uses the LOBPCG 
algorithm.

The function returns a tuple `(λ, X)`, where `λ` is a vector of eigenvalues and 
`X` is a matrix where each column represents an eigenvector.
    
Keyword arguments `kwargs...` can be used to pass additional parameters to the LOBPCG algorithm.
"""
function haml_eigs_default(H::SparseHamiltonian, t, X0::Matrix, kwargs...)
    res = lobpcg_hyper(H(t), X0; kwargs...)
    return (res.λ, res.X)
end

haml_eigs_default(H::SparseHamiltonian, t, ::Nothing) =
    eigen(H(t) |> Array |> Hermitian)

"""
$(TYPEDEF)

Defines a time independent Hamiltonian object with sparse matrices.

# Fields

$(FIELDS)
"""
struct ConstantSparseHamiltonian{T<:Number} <: AbstractHamiltonian{T}
    "Internal cache"
    u_cache::SparseMatrixCSC{T,Int}
    "Size"
    size::Tuple
end

function SparseHamiltonian(mat; unit=:h)
    mat = unit_scale(unit) * mat
    ConstantSparseHamiltonian(mat, size(mat))
end

isconstant(::ConstantSparseHamiltonian) = true
issparse(::ConstantSparseHamiltonian) = true

function (h::ConstantSparseHamiltonian)(::Real)
    h.u_cache
end

function update_cache!(cache, H::ConstantSparseHamiltonian, ::Any, ::Real)
    cache .= -1.0im * H.u_cache
end

function update_vectorized_cache!(cache, H::ConstantSparseHamiltonian, ::Any, ::Real)
    hmat = H.u_cache
    iden = sparse(I, size(H))
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end

function (h::ConstantSparseHamiltonian)(
    du,
    u::AbstractMatrix,
    ::Any,
    ::Real,
)
    H = h.u_cache
    du .= -1.0im * (H * u - u * H)
end

function Base.convert(S::Type{T}, H::ConstantSparseHamiltonian{M}) where {T<:Number,M}
    mat = convert.(S, H.u_cache)
    ConstantSparseHamiltonian{eltype(mat)}(mat, size(H))
end

function rotate(H::ConstantSparseHamiltonian, v)
    hsize = size(H)
    mat = v' * H.u_cache * v
    issparse(mat) ? ConstantSparseHamiltonian(mat, hsize) : ConstantDenseHamiltonian(mat, hsize)
end