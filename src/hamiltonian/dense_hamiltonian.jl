"""
$(TYPEDEF)

Defines a time dependent Hamiltonian object with dense Matrices.

# Fields

$(FIELDS)
"""
struct DenseHamiltonian{T<:Number} <: AbstractDenseHamiltonian{T}
    " List of time dependent functions "
    f
    " List of constant matrices "
    m::Vector{Matrix{T}}
    """Internal cache"""
    u_cache::Matrix{T}
    """Size"""
    size
end


"""
    function DenseHamiltonian(funcs, mats; unit = :h)

Constructor of DenseHamiltonian object. `funcs` and `mats` are a list of time dependent functions and the corresponding matrices. `unit` is the unit one -- `:h` or `:ħ`. The `mats` will be scaled by ``2π`` if unit is `:h`.
"""
function DenseHamiltonian(funcs, mats; unit = :h)
    if any((x) -> size(x) != size(mats[1]), mats)
        throw(ArgumentError("Matrices in the list do not have the same size."))
    end
    if is_complex(funcs, mats)
        mats = complex.(mats)
    end
    cache = similar(sum(mats))
    DenseHamiltonian(funcs, unit_scale(unit) * mats, cache, size(mats[1]))
end


"""
    function (h::DenseHamiltonian)(s::Real)

Calling the Hamiltonian returns the value ``2πH(s)``. The argument `s` is in the unitless time. The returned matrix is in angular frequency.
"""
function (h::DenseHamiltonian)(s::Real)
    fill!(h.u_cache, 0.0)
    for (f, m) in zip(h.f, h.m)
        axpy!(f(s), m, h.u_cache)
    end
    h.u_cache
end


@inline function (h::DenseHamiltonian)(tf::Real, t::Real)
    hmat = h(t)
    lmul!(tf, hmat)
end


@inline function (h::DenseHamiltonian)(tf::UnitTime, t::Real)
    hmat = h(t / tf)
end


# update_func interface for DiffEqOperators
function update_cache!(cache, H::DenseHamiltonian, tf::Real, s::Real)
    fill!(cache, 0.0)
    for (f, m) in zip(H.f, H.m)
        axpy!(f(s), m, cache)
    end
    lmul!(-1.0im * tf, cache)
end

update_cache!(cache, H::DenseHamiltonian, tf::UnitTime, t::Real) =
    update_cache!(cache, H, 1.0, t / tf)


function update_vectorized_cache!(cache, H::DenseHamiltonian, tf, t::Real)
    hmat = H(tf, t)
    iden = Matrix{eltype(H)}(I, size(H))
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end


function get_cache(H::DenseHamiltonian, vectorize)
    if vectorize == false
        get_cache(H)
    else
        get_cache(H) ⊗ Matrix{eltype(H)}(I, size(H))
    end
end


function (h::DenseHamiltonian)(du, u::Matrix{T}, p::Real, t::Real) where {T<:Complex}
    fill!(du, 0.0 + 0.0im)
    H = h(t)
    gemm!('N', 'N', -1.0im * p, H, u, 1.0 + 0.0im, du)
    gemm!('N', 'N', 1.0im * p, u, H, 1.0 + 0.0im, du)
end

(h::DenseHamiltonian)(du, u, tf::UnitTime, t::Real) = h(du, u, 1.0, t/tf)


"""
    function eigen_decomp(h::AbstractDenseHamiltonian, t; level = 2) -> (w, v)

Calculate the eigen value decomposition of the Hamiltonian `h` at time `t`. Keyword argument `level` specifies the number of levels to keep in the output. `w` is a vector of eigenvalues and `v` is a matrix of the eigenvectors in the columns. (The `k`th eigenvector can be obtained from the slice `v[:, k]`.) `w` will be in unit of `GHz`.
"""
function eigen_decomp(h::AbstractDenseHamiltonian, t; level = 2, kwargs...)
    H = h(t)
    w, v = eigen!(Hermitian(H), 1:level)
    lmul!(1 / 2 / π, w), v
end


function Base.convert(S::Type{T}, H::DenseHamiltonian) where {T<:Complex}
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, S)
    DenseHamiltonian(H.f, mats, cache, size(H))
end


function Base.convert(S::Type{T}, H::DenseHamiltonian) where {T<:Real}
    f_val = sum((x) -> x(0.0), H.f)
    if !(typeof(f_val) <: Real)
        throw(TypeError(:convert, "H.f", Real, typeof(f_val)))
    end
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, S)
    DenseHamiltonian(H.f, mats, cache, size(H))
end
