struct DiagonalOperator{T<:Real}
    ω_vec::Tuple
    u_cache::Vector{T}
end


struct GeometricOperator{T<:Number}
    funcs::Tuple
    u_cache::Matrix{T}
end


"""
$(TYPEDEF)

Defines a time dependent Hamiltonian in adiabatic frame.

# Fields

$(FIELDS)
"""
struct AdiabaticFrameHamiltonian{T} <: AbstractDenseHamiltonian{T}
    """Geometric part"""
    geometric::GeometricOperator
    """Adiabatic part"""
    diagonal::DiagonalOperator
    """Size of the Hamiltonian"""
    size
end


"""
    function AdiabaticFrameHamiltonian(ωfuns, geofuns)

Constructor of adiabatic frame Hamiltonian. `ωfuns` is a 1-D array of functions which specify the eigen energies (in `GHz`) of the Hamiltonian. `geofuns` is a 1-D array of functions which specifies the geometric phases of the Hamiltonian. `geofuns` can be thought as a flattened lower triangular matrix (without diagonal elements) in column-major order.
"""
function AdiabaticFrameHamiltonian(ωfuns, geofuns)
    diag_op = DiagonalOperator(ωfuns)
    geo_op = GeometricOperator(geofuns)
    op_size = (length(ωfuns), length(ωfuns))
    T = complex(typeof(geo_op).parameters[1])
    AdiabaticFrameHamiltonian{T}(geo_op, diag_op, op_size)
end


function update_cache!(cache, H::AdiabaticFrameHamiltonian, tf, t::Real)
    hmat = H(tf, t)
    cache .= -1.0im * hmat
end


function get_cache(H::AdiabaticFrameHamiltonian{T}) where T
    zeros(T, size(H))
end


function (H::AdiabaticFrameHamiltonian)(tf::Real, t::Real)
    ω = 2π * tf * H.diagonal(t)
    off = H.geometric(t)
    ω + off
end


function (H::AdiabaticFrameHamiltonian)(tf::UnitTime, t::Real)
    ω = 2π * H.diagonal(t / tf)
    off = H.geometric(t / tf) / tf
    ω + off
end


"""
    function evaluate(H::AdiabaticFrameHamiltonian, s, tf)

Evaluate the adiabatic frame Hamiltonian at (unitless) time `s`, with total annealing time `tf` (in the unit of ``ns``). The final result is given in unit of ``GHz``.
"""
function evaluate(H::AdiabaticFrameHamiltonian, s, tf)
    ω = H.diagonal(s)
    off = H.geometric(s) / tf
    ω + off
end


function (h::AdiabaticFrameHamiltonian)(
    du::Matrix{T},
    u::Matrix{T},
    tf::Real,
    t::Real
) where T <: Number
    ω = h.diagonal(t)
    du .= -2.0im * π * tf * (ω * u - u * ω)
    G = h.geometric(t)
    du .+= -1.0im * (G * u - u * G)
end


function (h::AdiabaticFrameHamiltonian)(
    du::Matrix{T},
    u::Matrix{T},
    tf::UnitTime,
    t::Real
) where T <: Number
    s = t / tf
    ω = h.diagonal(s)
    du .= -2.0im * π * (ω * u - u * ω)
    G = h.geometric(s)
    du .+= -1.0im * (G * u - u * G) / tf
end


function ω_matrix(H::AdiabaticFrameHamiltonian, lvl)
    ω = 2π * H.diagonal.u_cache[1:lvl]
    ω' .- ω
end


function ω_matrix_RWA(H::AdiabaticFrameHamiltonian, tf, t, lvl)
    ω = 2π * H.diagonal(t)
    off = H.geometric(t) / tf
    ω + off
    eigen!(Hermitian(ω + off), 1:lvl)
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


#TODO The following functions will be deprecated
function (h::AdiabaticFrameHamiltonian)(
    du::Vector{T},
    u::Vector{T},
    tf::Real,
    t::Real
) where T
    ω = h.diagonal(t)
    du .= -2.0im * π * tf * ω * u
    G = h.geometric(t)
    du .+= -1.0im * G * u
end


function (h::AdiabaticFrameHamiltonian)(
    du::Vector{T},
    u::Vector{T},
    tf::UnitTime,
    t::Real
) where T <: Number
    ω = h.diagonal(t / tf)
    du .= -2.0im * π * ω * u
    G = h.geometric(t / tf)
    du .+= -1.0im / tf * G * u
end
