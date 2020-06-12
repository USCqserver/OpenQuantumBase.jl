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
    size::Any
end


"""
    function SparseHamiltonian(funcs, mats; unit=:h)

Constructor of SparseHamiltonian object. `funcs` and `mats` are a list of time dependent functions and the corresponding matrices. `unit` is the unit one -- `:h` or `:ħ`. The `mats` will be scaled by ``2π`` is unit is `:h`.
"""
function SparseHamiltonian(funcs, mats; unit = :h)
    if any((x) -> size(x) != size(mats[1]), mats)
        throw(ArgumentError("Matrices in the list do not have the same size."))
    end
    if is_complex(funcs, mats)
        mats = complex.(mats)
    end
    cache = similar(sum(mats))
    fill!(cache, 0.0)
    SparseHamiltonian(funcs, unit_scale(unit) * mats, cache, size(mats[1]))
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


@inline function (h::SparseHamiltonian)(tf::Real, t::Real)
    hmat = h(t)
    lmul!(tf, hmat)
end


@inline function (h::SparseHamiltonian)(tf::UnitTime, t::Real)
    hmat = h(t / tf)
end


# update_func interface for DiffEqOperators
function update_cache!(cache, H::SparseHamiltonian, tf::Real, s::Real)
    fill!(cache, 0.0)
    for (f, m) in zip(H.f, H.m)
        cache .+= -1.0im * tf * f(s) * m
    end
end

update_cache!(cache, H::SparseHamiltonian, tf::UnitTime, t::Real) =
    update_cache!(cache, H, 1.0, t / tf)


function update_vectorized_cache!(cache, H::SparseHamiltonian, tf, t::Real)
    hmat = H(tf, t)
    iden = sparse(I, size(H))
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end


function get_cache(H::SparseHamiltonian, vectorize)
    if vectorize == false
        get_cache(H)
    else
        h_cache = similar(get_cache(H))
        fill!(h_cache, 1.0)
        h_cache ⊗ sparse(I, size(H)) + sparse(I, size(H)) ⊗ h_cache
    end
end


function (h::SparseHamiltonian)(
    du,
    u::Matrix{T},
    tf::Real,
    t::Real,
) where {T<:Number}
    H = h(t)
    du .= -1.0im * tf * (H * u - u * H)
end

(h::SparseHamiltonian)(du, u, tf::UnitTime, t::Real) = h(du, u, 1.0, t / tf)


function Base.convert(S::Type{T}, H::SparseHamiltonian) where {T<:Complex}
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, S)
    SparseHamiltonian(H.f, mats, cache, size(H))
end


function Base.convert(S::Type{T}, H::SparseHamiltonian) where {T<:Real}
    f_val = sum((x) -> x(0.0), H.f)
    if !(typeof(f_val) <: Real)
        throw(TypeError(:convert, "H.f", Real, typeof(f_val)))
    end
    mats = [convert.(S, x) for x in H.m]
    cache = similar(H.u_cache, S)
    SparseHamiltonian(H.f, mats, cache, size(H))
end
