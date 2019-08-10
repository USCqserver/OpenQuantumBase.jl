struct DiagonalOperator{T<:Real}
    ω_vec::Tuple
    u_cache::Vector{T}
end

function DiagonalOperator(funcs...)
    num_type = typeof(funcs[1](0.0))
    DiagonalOperator{num_type}(funcs, zeros(num_type, length(funcs)))
end

function DiagonalOperator(funcs::Vector{T}) where T
    DiagonalOperator(funcs...)
end

function (D::DiagonalOperator)(t)
    for i in eachindex(D.ω_vec)
        D.u_cache[i] = D.ω_vec[i](t)
    end
    Diagonal(D.u_cache)
end

struct GeometricOperator{T<:Number}
    funcs::Tuple
    u_cache::Matrix{T}
end

function GeometricOperator(funcs...)
    dim = (sqrt(1 + 8 * length(funcs)) - 1) / 2
    if !isinteger(dim)
        throw(ArgumentError("Invalid input length."))
    else
        dim = Int(dim)
    end
    num_type = typeof(funcs[1](0.0))
    GeometricOperator{num_type}(funcs, zeros(num_type, dim + 1, dim + 1))
end

function GeometricOperator(funcs::Vector{T}) where T
    GeometricOperator(funcs...)
end

function (G::GeometricOperator)(t)
    len = size(G.u_cache, 1)
    for j = 1:len
        for i = (j+1):len
            G.u_cache[i, j] = G.funcs[i-1+(j-1)*len](t)
        end
    end
    Hermitian(G.u_cache, :L)
end

struct AdiabaticFrameHamiltonian{T} <: AbstractDenseHamiltonian{T}
    geometric::GeometricOperator
    diagonal::DiagonalOperator{T}
    size
end


function AdiabaticFrameHamiltonian(ωfuns, geofuns)
    diag_op = DiagonalOperator(ωfuns)
    geo_op = GeometricOperator(geofuns)
    op_size = (length(ωfuns), length(ωfuns))
    AdiabaticFrameHamiltonian(geo_op, diag_op, op_size)
end


function (H::AdiabaticFrameHamiltonian)(tf::Real, t::Real)
    ω = 2π * H.diagonal(t)
    off = 2π * H.geometric(t) / tf
    ω + off
end


function evaluate(H::AdiabaticFrameHamiltonian, t, tf)
    H.(tf, t) / 2 / π
end


function (h::AdiabaticFrameHamiltonian)(du, u::Vector{T}, tf::Real, t::Real) where T
    ω = h.diagonal(t)
    du .= -2.0im * π * tf * ω * u
    G = h.geometric(t)
    du .+= -2.0im * π * G * u
end


function (h::AdiabaticFrameHamiltonian)(du, u::Vector{T}, tf::UnitTime, t::Real) where T
    ω = h.diagonal(t/tf)
    du .= -2.0im * π * ω * u
    G = h.geometric(t/tf)
    du .+= -2.0im * π / tf * G * u
end


function (h::AdiabaticFrameHamiltonian)(du, u, tf::Real, t::Real)
    ω = h.diagonal(t)
    du .= -2.0im * π * tf * (ω * u - u * ω)
    G = h.geometric(t)
    du .+= -2.0im * π * (G * u - u * G)
end


function ω_matrix(H::AdiabaticFrameHamiltonian)
    ω = 2π * H.diagonal.u_cache
    ω' .- ω
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
