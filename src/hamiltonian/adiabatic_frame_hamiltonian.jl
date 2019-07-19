struct AdiabaticFrameHamiltonian{T} <: AbstractDenseHamiltonian{T}
    geometric::LinearOperator{T}
    adiabatic::LinearOperator{T}
    u_cache::Array{T, 2}
    ω_cache::Array{T, 1}
end

function AdiabaticFrameHamiltonian(hfuncs, hops, gfuncs, gops)
    geometric_op = AffineOperator(gfuncs, gops)
    adiabatic_op = AffineOperator(hfuncs, hops)
    AdiabaticFrameHamiltonian(geometric_op, adiabatic_op,
    zeros(eltype(hops[1]), size(hops[1])),
    zeros(eltype(hops[1]), size(hops[1], 1)))
end

function (h::AdiabaticFrameHamiltonian)(tf::Real, t::Real)
    fill!(h.u_cache, 0.0)
    h.adiabatic(h.u_cache, tf, t)
    h.geometric(h.u_cache, t)
    h.u_cache
end

function (h::AdiabaticFrameHamiltonian)(du, u, tf::Real, t::Real)
    fill!(h.u_cache, 0.0)
    h.adiabatic(h.u_cache, t)
    h.ω_cache .= diag(h.u_cache)
    lmul!(tf, h.u_cache)
    h.geometric(h.u_cache, t)
    gemm!('N', 'N', -1.0im, h.u_cache, u, 1.0+0.0im, du)
    gemm!('N', 'N', 1.0im, u, h.u_cache, 1.0+0.0im, du)
end

function ω_matrix(H::AdiabaticFrameHamiltonian)
    H.ω_cache' .- H.ω_cache
end

# function (h::AdiabaticFrameHamiltonian)(du, u::StateMachineDensityMatrix, p::AdiabaticFramePauseControl, t::Real)
#     s = p.annealing_parameter[u.state](t)
#     fill!(h.u_cache, 0.0)
#     h.adiabatic(h.u_cache, s)
#     h.ω_cache .= diag(h.u_cache)
#     lmul!(p.tf, h.u_cache)
#     h.geometric(h.u_cache, p.geometric_scaling[u.state], s)
#     gemm!('N', 'N', -1.0im, h.u_cache, u.x, 1.0+0.0im, du.x)
#     gemm!('N', 'N', 1.0im, u.x, h.u_cache, 1.0+0.0im, du.x)
# end
