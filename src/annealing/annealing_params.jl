"""
$(TYPEDEF)

Base for types defining [parametrized functions](http://docs.juliadiffeq.org/latest/tutorials/ode_example.html) for annealing ODEs.
"""
abstract type AbstractAnnealingParams end

mutable struct AnnealingParams <: AbstractAnnealingParams
    H::AbstractHamiltonian
    tf
    opensys
    control
end

function AnnealingParams(H, tf; opensys=nothing, control=nothing)
    AnnealingParams(H, tf, opensys, control)
end
