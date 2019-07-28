"""
$(TYPEDEF)

Defines a time dependent Hamiltonian object with sparse Matrices. All the values in the input is assumed to have the unit of `GHz`. An additional ``2π`` factor will be multiplied to each matrices when constructing the object.

# Fields

$(FIELDS)
"""
struct SparseHamiltonian{T} <: AbstractSparseHamiltonian{T}
    """Linear Operator Implementation"""
    op::LinearOperatorSparse{T}
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
    cache = spzeros(eltype(mats[1]), size(mats[1])...)
    operator = AffineOperatorSparse(funcs, 2π * mats)
    SparseHamiltonian(operator, cache, size(mats[1]))
end

"""
    function (h::SparseHamiltonian)(t::Real)

Calling the Hamiltonian returns the value ``2πH(t)``.
"""
function (h::SparseHamiltonian)(t::Real)
    fill!(h.u_cache, 0.0)
    h.op(h.u_cache, t)
    h.u_cache
end


function (h::SparseHamiltonian)(du, u, p::Real, t::Real)
    H = h(t)
    du .+= -1.0im * p * (H * u - u * H)
end


function p_copy(h::SparseHamiltonian)
    SparseHamiltonian(h.op, spzeros(eltype(h.u_cache), size(h.u_cache)...), h.size)
end


"""
    function eigen_decomp(h::AbstractSparseHamiltonian, t; level = 2) -> (w, v)

Calculate the eigen value decomposition of the Hamiltonian `h` at time `t`. Keyword argument `level` specifies the number of levels to keep in the output. `w` is a vector of eigenvalues and `v` is a matrix of the eigenvectors in the columns. (The `k`th eigenvector can be obtained from the slice `w[:, k]`.) `w` will be in unit of `GHz`. [Arpack.jl](https://julialinearalgebra.github.io/Arpack.jl/stable/) is used internally for solving eigensystems of sparse matrices. Any keyword arguments of `eigs` function is supported here.
"""
function eigen_decomp(h::AbstractSparseHamiltonian, t; level = 2, kwargs...)
    H = h(t)
    w, v = eigs(H; nev = level, which = :SR, kwargs...)
    w/2/π, v
end

@inline function ode_eigen_decomp(h::AbstractSparseHamiltonian)
    eigs(h.u_cache; nev = h.size[1]-1, which = :SR)
end
