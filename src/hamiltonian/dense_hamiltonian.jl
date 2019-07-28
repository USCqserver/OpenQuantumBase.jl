"""
$(TYPEDEF)

Defines a time dependent Hamiltonian object with dense Matrices. All the values in the input is assumed to have the unit of `GHz`. An additional ``2π`` factor will be multiplied to each matrices when constructing the object.

# Fields

$(FIELDS)
"""
struct DenseHamiltonian{T <: Complex} <: AbstractDenseHamiltonian{T}
    """Linear Operator Implementation"""
    op::LinearOperator{T}
    """Internal cache"""
    u_cache::Matrix{T}
    """Size"""
    size
end


"""
    function DenseHamiltonian(funcs, mats)

Constructor of DenseHamiltonian object. `funcs` and `mats` are a list of time dependent functions and the corresponding matrices.
"""
function DenseHamiltonian(funcs, mats)
    cache = zeros(eltype(mats[1]), size(mats[1]))
    # the matrices are scaling by 2π
    operator = AffineOperator(funcs, 2π*mats)
    DenseHamiltonian(operator, cache, size(mats[1]))
end


"""
    function (h::DenseHamiltonian)(t::Real)

Calling the Hamiltonian returns the value ``2πH(t)``.
"""
function (h::DenseHamiltonian)(t::Real)
    fill!(h.u_cache, 0.0)
    h.op(h.u_cache, t)
    h.u_cache
end


function (h::DenseHamiltonian)(du, u::Matrix{T}, p::Real, t::Real) where T<:Complex
    H = h(t)
    gemm!('N', 'N', -1.0im * p, H, u, 1.0 + 0.0im, du)
    gemm!('N', 'N', 1.0im * p, u, H, 1.0 + 0.0im, du)
end


function (h::DenseHamiltonian)(du, u::Vector{T}, p::Real, t::Real) where T<:Complex
    H = h(t)
    mul!(du, -1.0im * p * H, u)
end


function p_copy(h::DenseHamiltonian)
    DenseHamiltonian(h.op, zeros(eltype(h.u_cache), size(h.u_cache)), h.size)
end


"""
    function eigen_decomp(h::AbstractDenseHamiltonian, t; level = 2) -> (w, v)

Calculate the eigen value decomposition of the Hamiltonian `h` at time `t`. Keyword argument `level` specifies the number of levels to keep in the output. `w` is a vector of eigenvalues and `v` is a matrix of the eigenvectors in the columns. (The `k`th eigenvector can be obtained from the slice `w[:, k]`.) `w` will be in unit of `GHz`.
"""
function eigen_decomp(h::AbstractDenseHamiltonian, t; level = 2)
    H = h(t)
    w, v = eigen!(Hermitian(H))
    w[1:level]/2/π, v[:, 1:level]
end

@inline function ode_eigen_decomp(h::AbstractDenseHamiltonian)
    eigen!(Hermitian(h.u_cache))
end
