"""
$(TYPEDEF)

Defines a time dependent Hamiltonian object with sparse Matrices.

# Fields

$(FIELDS)
"""
struct SparseHamiltonian{T<:Number} <: AbstractSparseHamiltonian{T}
    " List of time dependent functions "
    f::Any
    " List of constant matrices "
    m::Vector{SparseMatrixCSC{T,Int}}
    """Internal cache"""
    u_cache::SparseMatrixCSC{T,Int}
    """Size"""
    size::Tuple
    """Eigen decomposition routine"""
    EIGS::Any
end

"""
$(SIGNATURES)

Constructor of SparseHamiltonian object. `funcs` and `mats` are a list of time dependent functions and the corresponding matrices. `unit` is the unit one -- `:h` or `:ħ`. The `mats` will be scaled by ``2π`` is unit is `:h`.

`EIGS` is the initializer for the eigen decomposition routine for the Hamiltonian. It should return a function of signature: `(H, s, lvl) -> (w, v)`. The default method `EIGEN_DEFAULT` will use `LAPACK` routine.
"""
function SparseHamiltonian(funcs, mats; unit = :h, EIGS = EIGEN_DEFAULT)
    if any((x) -> size(x) != size(mats[1]), mats)
        throw(ArgumentError("Matrices in the list do not have the same size."))
    end
    if is_complex(funcs, mats)
        mats = complex.(mats)
    end
    cache = similar(sum(mats))
    fill!(cache, 0.0)
    mats = unit_scale(unit) * mats
    EIGS = EIGS(cache)
    SparseHamiltonian(funcs, mats, cache, size(mats[1]), EIGS)
end


"""
    function (h::SparseHamiltonian)(t::Real)

Calling the Hamiltonian returns the value ``2πH(t)``.
"""
function (h::SparseHamiltonian)(t::Real)
    fill!(h.u_cache, 0.0)
    for (f, m) in zip(h.f, h.m)
        h.u_cache .+= f(t) * m
    end
    h.u_cache
end

# update_func interface for DiffEqOperators
function update_cache!(cache, H::SparseHamiltonian, tf, s::Real)
    fill!(cache, 0.0)
    for (f, m) in zip(H.f, H.m)
        cache .+= -1.0im * f(s) * m
    end
end

function update_vectorized_cache!(cache, H::SparseHamiltonian, tf, s::Real)
    hmat = H(s)
    iden = sparse(I, size(H))
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end

function (h::SparseHamiltonian)(
    du,
    u,
    p,
    s::Real,
) where {T<:Number}
    H = h(s)
    du .= -1.0im * (H * u - u * H)
end

function Base.convert(S::Type{T}, H::SparseHamiltonian{M}) where {T<:Complex, M}
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, complex{M})
    SparseHamiltonian(H.f, mats, cache, size(H))
end

function Base.convert(S::Type{T}, H::SparseHamiltonian{M}) where {T<:Real, M}
    f_val = sum((x) -> x(0.0), H.f)
    if !(typeof(f_val) <: Real)
        throw(TypeError(:convert, "H.f", Real, typeof(f_val)))
    end
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, real(M))
    SparseHamiltonian(H.f, mats, cache, size(H), H.EIGS)
end
