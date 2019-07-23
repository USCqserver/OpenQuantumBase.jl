struct SparseHamiltonian{T} <: AbstractSparseHamiltonian{T}
    op::LinearOperatorSparse{T}
    u_cache::SparseMatrixCSC{T,Int}
    size
end

function SparseHamiltonian(funcs, mats)
    cache = spzeros(eltype(mats[1]), size(mats[1])...)
    operator = AffineOperatorSparse(funcs, 2π * mats)
    SparseHamiltonian(operator, cache, size(mats[1]))
end

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

function eigen_decomp(h::AbstractSparseHamiltonian, t; level = 2, kwargs...)
    H = h(t)
    w, v = eigs(H; nev = level, which = :SR, kwargs...)
end
