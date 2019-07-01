struct Hamiltonian{T} <: AbstractHamiltonian{T}
    op::LinearOperator{T}
    u_cache::Array{T, 2}
end

struct HamiltonianSparse{T} <: AbstractHamiltonian{T}
    op::LinearOperator{T}
    u_cache::SparseMatrixCSC{T, Int64}
end

function (h::Hamiltonian)(t::Real)
    fill!(h.u_cache, 0.0)
    h.op(h.u_cache, t)
    h.u_cache
end

function (h::Hamiltonian)(du, u, t::Real)
    H = h(t)
    gemm!('N', 'N', -1.0im, H, u, 1.0+0.0im, du)
    gemm!('N', 'N', 1.0im, u, H, 1.0+0.0im, du)
end

function (h::HamiltonianSparse)(t::Real)
    fill!(h.u_cache, 0.0)
    h.op(h.u_cache, t)
    h.u_cache
end

function (h::HamiltonianSparse)(du, u, t::Real)
    H = h(t)
    axpy!(-1.0im, H*u, du)
    axpy!(1.0im, u*H, du)
end

function scale!(h::Union{Hamiltonian, HamiltonianSparse}, a::Real)
    h.op.m .*= a
end
