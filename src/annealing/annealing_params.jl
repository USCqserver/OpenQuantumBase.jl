"""
$(TYPEDEF)

Base for types defining [parametrized functions](http://docs.juliadiffeq.org/latest/tutorials/ode_example.html) for annealing ODEs.
"""
abstract type AbstractAnnealingParams end

struct AnnealingParams{T<:Union{AbstractFloat, UnitTime}} <: AbstractAnnealingParams
    H::AbstractHamiltonian
    tf::T
    opensys
    control
end

function set_tf(P::AnnealingParams{T}, tf::Real) where T<:AbstractFloat
    AnnealingParams(P.H, float(tf), P.opensys, P.control)
end

function set_tf(P::AnnealingParams{T}, tf::Real) where T<:UnitTime
    AnnealingParams(P.H, UnitTime(tf), P.opensys, P.control)
end

function AnnealingParams(H, tf::T; opensys=nothing, control=nothing) where T<:Number
    AnnealingParams(H, float(tf), opensys, control)
end

function AnnealingParams(H, tf; opensys=nothing, control=nothing)
    AnnealingParams(H, tf, opensys, control)
end
