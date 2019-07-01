mutable struct ControlDensityMatrix{T} <: DEDataMatrix{T}
    x::Array{T,2}
    stage::Int
end

mutable struct AdiabaticFramePiecewiseControl <: AnnealingControl
    tf::Float64
    sf::Float64
    stops
    annealing_parameter
    geometric_scaling
end

function pausing_annealing_parameter(sp, sd)
    sf = 1 + sd
    stops = [sp, sp+sd]
    annealing_parameter = [(x)->x ,(x)->sp, (x)->x-sp]
    geometric_scaling = [1.0, 0.0, 1.0]
    AdiabaticFramePiecewiseControl(1.0, sf, stops, annealing_parameter, geometric_scaling)
end
