"""
$(TYPEDEF)

Defines a time dependent Hamiltonian object with sparse Matrices. All the values in the input is assumed to have the unit of `GHz`. An additional ``2π`` factor will be multiplied to each matrices when constructing the object.

# Fields

$(FIELDS)
"""
struct SparseHamiltonian{T} <: AbstractSparseHamiltonian{T}
    " List of time dependent functions "
    f
    " List of constant matrices "
    m::Vector{SparseMatrixCSC{T,Int}}
    """Internal cache"""
    u_cache::SparseMatrixCSC{T,Int}
    """Size"""
    size
end


"""
    function SparseHamiltonian(funcs, mats)

Constructor of SparseHamiltonian object. `funcs` and `mats` are a list of time dependent functions and the corresponding matrices.
"""
function SparseHamiltonian(funcs, mats)
    if !all((x) -> size(x) == size(mats[1]), mats)
        throw(ArgumentError("Matrices in the list do not have the same size."))
    end
    cache = sum(mats)
    fill!(cache, 0.0 + 0.0im)
    SparseHamiltonian(funcs, 2π * mats, cache, size(mats[1]))
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


function (h::SparseHamiltonian)(tf::Real, t::Real)
    hmat = h(t)
    lmul!(tf, hmat)
end


function (h::SparseHamiltonian)(tf::UnitTime, t::Real)
    hmat = h(t / tf)
end


function (h::SparseHamiltonian)(du, u::Matrix{T}, tf::Real, t::Real) where T <: Number
    H = h(t)
    du .= -1.0im * tf * (H * u - u * H)
end


function (h::SparseHamiltonian)(du, u::Matrix{T}, tf::UnitTime, t::Real) where T <: Number
    H = h(t / tf)
    du .= -1.0im * (H * u - u * H)
end


function (h::SparseHamiltonian)(du, u::Vector{T}, tf::Real, t::Real) where T <: Number
    H = h(t)
    mul!(du, -1.0im * tf * H, u)
end


function (h::SparseHamiltonian)(du, u::Vector{T}, tf::UnitTime, t::Real) where T <: Number
    H = h(t/tf)
    mul!(du, -1.0im * H, u)
end


"""
    function eigen_decomp(h::AbstractSparseHamiltonian, t; level = 2) -> (w, v)

Calculate the eigen value decomposition of the Hamiltonian `h` at time `t`. Keyword argument `level` specifies the number of levels to keep in the output. `w` is a vector of eigenvalues and `v` is a matrix of the eigenvectors in the columns. (The `k`th eigenvector can be obtained from the slice `w[:, k]`.) `w` will be in unit of `GHz`. [Arpack.jl](https://julialinearalgebra.github.io/Arpack.jl/stable/) is used internally for solving eigensystems of sparse matrices. Any keyword arguments of `eigs` function is supported here.
"""
function eigen_decomp(h::AbstractSparseHamiltonian, t; level = 2, kwargs...)
    H = h(t)
    w, v = eigs(H; nev = level, which = :SR, kwargs...)
    lmul!(1 / 2 / π, w), v
end


function ode_eigen_decomp(h::AbstractSparseHamiltonian, lvl::Integer)
    w, v = eigs(h.u_cache; nev = lvl, which = :SR)
    w, v
end


function p_copy(h::SparseHamiltonian)
    SparseHamiltonian(h.f, h.m, spzeros(eltype(h.u_cache), size(h.u_cache)...), h.size)
end
