macro update!(res, t, A)
    return quote
        local lA = $(esc(A))
        local lt = $(esc(t))
        local lres = $(esc(res))
        for (f, m) in zip(lA.f, lA.m)
            axpy!(f(lt), m, lres)
        end
    end
end

macro update!(res, t, scale, A)
    return quote
        local lA = $(esc(A))
        local lt = $(esc(t))
        local lres = $(esc(res))
        local ls = $(esc(scale))
        for (f, m) in zip(lA.f, lA.m)
            axpy!(ls*f(lt), m, lres)
        end
    end
end

struct AffineOperator{T<:Number} <: LinearOperator{T}
    " List of time dependent functions "
    f
    " List of constant matrices "
    m::Union{Array{Array{T, 2}, 1}, Array{SparseMatrixCSC{T, Int64}, 1}}
end

struct Hamiltonian{T} <: AbstractHamiltonian{T}
    op::LinearOperator{T}
    u_cache::Array{T, 2}
end

struct HamiltonianSparse{T} <: AbstractHamiltonian{T}
    op::LinearOperator{T}
    u_cache::SparseMatrixCSC{T, Int64}
end

function hamiltonian_factory(funcs, ops)
    operator = AffineOperator(funcs, ops)
    if issparse(ops[1])
        cache = spzeros(eltype(ops[1]), size(ops[1])...)
        HamiltonianSparse(operator, cache)
    else
        cache = zeros(eltype(ops[1]), size(ops[1]))
        Hamiltonian(operator, cache)
    end
end

function (h::Hamiltonian)(t::Real)
    fill!(h.u_cache, 0.0)
    @update!(h.u_cache, t, h.op)
    h.u_cache
end

function (h::Hamiltonian)(du, u, t::Real)
    H = h(t)
    gemm!('N', 'N', -1.0im, H, u, 1.0+0.0im, du)
    gemm!('N', 'N', 1.0im, u, H, 1.0+0.0im, du)
end

function (h::HamiltonianSparse)(t::Real)
    fill!(h.u_cache, 0.0)
    @update!(h.u_cache, t, h.op)
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

struct AdiabaticFrameHamiltonian{T} <: AbstractHamiltonian{T}
    geometric::LinearOperator{T}
    adiabatic::LinearOperator{T}
    u_cache::Array{T, 2}
end

function hamiltonian_factory(hfuncs, hops, gfuncs, gops)
    geometric_op = AffineOperator(gfuncs, gops)
    adiabatic_op = AffineOperator(hfuncs, hops)
    AdiabaticFrameHamiltonian(geometric_op, adiabatic_op, zeros(eltype(hops[1]), size(hops[1])))
end

function (h::AdiabaticFrameHamiltonian)(t::Real)
    fill!(h.u_cache, 0.0)
    @update!(h.u_cache, t, h.adiabatic)
    @update!(h.u_cache, t, h.geometric)
    h.u_cache
end

function (h::AdiabaticFrameHamiltonian)(tf::Number, t::Real)
    fill!(h.u_cache, 0.0)
    @update!(h.u_cache, t, tf, h.adiabatic)
    @update!(h.u_cache, t, h.geometric)
    h.u_cache
end

function (h::AdiabaticFrameHamiltonian)(p::AnnealingControl, t::Real)
    fill!(h.u_cache, 0.0)
    @update!(h.u_cache, t, p.tf*p.adiabatic_scaling[1], h.adiabatic)
    @update!(h.u_cache, t, p.geometric_scaling[1], h.geometric)
    gemm!('N', 'N', -1.0im, h.u_cache, u, 1.0+0.0im, du)
    gemm!('N', 'N', 1.0im, u, h.u_cache, 1.0+0.0im, du)
    h.u_cache
end

function (h::AdiabaticFrameHamiltonian)(du, u, tf::Number, t::Real)
    fill!(h.u_cache, 0.0)
    @update!(h.u_cache, t, tf, h.adiabatic)
    @update!(h.u_cache, t, h.geometric)
    gemm!('N', 'N', -1.0im, h.u_cache, u, 1.0+0.0im, du)
    gemm!('N', 'N', 1.0im, u, h.u_cache, 1.0+0.0im, du)
end

function (h::AdiabaticFrameHamiltonian)(du, u, p::AnnealingControl, t::Real)
    fill!(h.u_cache, 0.0)
    @update!(h.u_cache, t, p.tf*p.adiabatic_scaling[1], h.adiabatic)
    @update!(h.u_cache, t, p.geometric_scaling[1], h.geometric)
    gemm!('N', 'N', -1.0im, h.u_cache, u, 1.0+0.0im, du)
    gemm!('N', 'N', 1.0im, u, h.u_cache, 1.0+0.0im, du)
end

function ω_matrix(H::AdiabaticFrameHamiltonian)
    ω = diag(H.u_cache)
    ω' .- ω
end

function scale!(h::AdiabaticFrameHamiltonian, a, type)
    if type == "geometric"
        h.geometric.m .*= a
    else
        h.adiabatic.m .*= a
    end
end
