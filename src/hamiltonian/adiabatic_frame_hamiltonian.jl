abstract type AbstractDiagonalOperator{T <: Number} end
abstract type AbstractGeometricOperator{T <: Number} end
Base.length(D::AbstractDiagonalOperator) = length(D.u_cache)
Base.eltype(::AbstractDiagonalOperator{T}) where {T} = T
Base.size(G::AbstractGeometricOperator, inds...) = size(G.u_cache, inds...)

struct DiagonalOperator{T} <: AbstractDiagonalOperator{T}
    ω_vec::Tuple
    u_cache::Vector{T}
end

function DiagonalOperator(funcs...)
    num_type = typeof(funcs[1](0.0))
    DiagonalOperator{num_type}(funcs, zeros(num_type, length(funcs)))
end

DiagonalOperator(funcs::Vector{T}) where {T} = DiagonalOperator(funcs...)

function (D::DiagonalOperator)(t)
    for i in eachindex(D.ω_vec)
        D.u_cache[i] = D.ω_vec[i](t)
    end
    Diagonal(D.u_cache)
end

struct DiagonalFunction{T} <: AbstractDiagonalOperator{T}
    func
    u_cache
end

function DiagonalFunction(func)
    tmp = func(0.0)
    DiagonalFunction{eltype(tmp)}(func, zero(tmp))
end

function (D::DiagonalFunction)(t)
    D.u_cache .= D.func(t)
    Diagonal(D.u_cache)
end

struct GeometricOperator{T} <: AbstractGeometricOperator{T}
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

GeometricOperator(funcs::Vector{T}) where {T} = GeometricOperator(funcs...)

function (G::GeometricOperator)(t)
    len = size(G.u_cache, 1)
    for j = 1:len
        for i = (j + 1):len
            G.u_cache[i, j] = G.funcs[i - 1 + (j - 1) * len](t)
        end
    end
    Hermitian(G.u_cache, :L)
end

struct ZeroGeometricOperator{T} <: AbstractGeometricOperator{T}
    size::Int
end

(G::ZeroGeometricOperator{T})(s) where {T} = Diagonal(zeros(T, G.size))

"""
$(TYPEDEF)

Defines a time dependent Hamiltonian in adiabatic frame.

# Fields

$(FIELDS)
"""
struct AdiabaticFrameHamiltonian{T} <: AbstractDenseHamiltonian{T}
    """Geometric part"""
    geometric::AbstractGeometricOperator
    """Adiabatic part"""
    diagonal::AbstractDiagonalOperator
    """Size of the Hamiltonian"""
    size::Any
end


"""
    function AdiabaticFrameHamiltonian(ωfuns, geofuns)

Constructor of adiabatic frame Hamiltonian. `ωfuns` is a 1-D array of functions which specify the eigen energies (in `GHz`) of the Hamiltonian. `geofuns` is a 1-D array of functions which specifies the geometric phases of the Hamiltonian. `geofuns` can be thought as a flattened lower triangular matrix (without diagonal elements) in column-major order.
"""
function AdiabaticFrameHamiltonian(D::AbstractDiagonalOperator{T}, G::AbstractGeometricOperator) where {T}
    l = length(D)
    AdiabaticFrameHamiltonian{complex(T)}(G, D, (l, l))
end

function AdiabaticFrameHamiltonian(ωfuns, geofuns)
    if isa(ωfuns, AbstractVector)
        D = DiagonalOperator(ωfuns)
    elseif isa(ωfuns, Function)
        D = DiagonalFunction(ωfuns)
    end
    l = length(D)
    T = eltype(D)

    if geofuns == nothing || isempty(geofuns)
        G = ZeroGeometricOperator{T}(l)
    else
        G = GeometricOperator(geofuns)
        if size(G, 1) != l
            error("Diagonal and geometric operators do not match in size.")
        end
    end
    
    AdiabaticFrameHamiltonian(D, G)
end

function (H::AdiabaticFrameHamiltonian)(tf::Real, s::Real)
    ω = 2π * H.diagonal(s)
    off = H.geometric(s) / tf
    ω + off
end

update_cache!(cache, H::AdiabaticFrameHamiltonian, tf, t::Real) = cache .= -1.0im * H(tf, t)
get_cache(H::AdiabaticFrameHamiltonian{T}) where {T} = zeros(T, size(H))

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
    s::Real,
) where {T <: Number}
    ω = h.diagonal(s)
    du .= -2.0im * π * (ω * u - u * ω)
    G = h.geometric(s)
    du .+= -1.0im * (G * u - u * G) / tf
end

function ω_matrix(H::AdiabaticFrameHamiltonian, lvl)
    ω = 2π * H.diagonal.u_cache[1:lvl]
    ω' .- ω
end
