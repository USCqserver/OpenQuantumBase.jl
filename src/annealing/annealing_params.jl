"""
$(TYPEDEF)

Base for types defining [parametrized functions](http://docs.juliadiffeq.org/latest/tutorials/ode_example.html) for annealing ODEs.
"""
abstract type AbstractAnnealingParams end

mutable struct AnnealingParams <: AbstractAnnealingParams
    H::AbstractHamiltonian
    tf::Float64
end
