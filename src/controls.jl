mutable struct StateMachineDensityMatrix{T} <: DEDataMatrix{T}
    x::Array{T,2}
    state::Int
end

mutable struct AdiabaticFramePauseControl <: AbstractPauseControl
    tf::Float64
    tstops
    annealing_parameter
    geometric_scaling
end

mutable struct PiecewiseAnnealingControl <: AbstractAnnealingControl
    tf::Float64
    tstops
end
