struct AdiabaticFrameHamiltonian{T} <: AbstractHamiltonian{T}
    geometric::LinearOperator{T}
    adiabatic::LinearOperator{T}
    u_cache::Array{T, 2}
    ω_cache::Array{T, 1}
end

function (h::AdiabaticFrameHamiltonian)(tf::Number, t::Real)
    fill!(h.u_cache, 0.0)
    h.adiabatic(h.u_cache, tf, t)
    h.geometric(h.u_cache, t)
    h.u_cache
end

function (h::AdiabaticFrameHamiltonian)(du, u, tf::Number, t::Real)
    fill!(h.u_cache, 0.0)
    h.adiabatic(h.u_cache, t)
    h.ω_cache .= diag(h.u_cache)
    lmul!(tf, h.u_cache)
    h.geometric(h.u_cache, t)
    gemm!('N', 'N', -1.0im, h.u_cache, u, 1.0+0.0im, du)
    gemm!('N', 'N', 1.0im, u, h.u_cache, 1.0+0.0im, du)
end

function (h::AdiabaticFrameHamiltonian)(du, u, p::AnnealingControl, t::Real)
    s = p.annealing_parameter[1](t)
    fill!(h.u_cache, 0.0)
    h.adiabatic(h.u_cache, s)
    h.ω_cache .= diag(h.u_cache)
    lmul!(p.tf, h.u_cache)
    h.geometric(h.u_cache, p.geometric_scaling[1], s)
    gemm!('N', 'N', -1.0im, h.u_cache, u, 1.0+0.0im, du)
    gemm!('N', 'N', 1.0im, u, h.u_cache, 1.0+0.0im, du)
end

function scale!(h::AdiabaticFrameHamiltonian, a, type)
    if type == "geometric"
        h.geometric.m .*= a
    else
        h.adiabatic.m .*= a
    end
end

function ω_matrix(H::AdiabaticFrameHamiltonian)
    H.ω_cache' .- H.ω_cache
end
