struct Annealing <: AbstractAnnealing
    ode_problem
    kwargs
end

function solve(annealing::Annealing, method, tf::Number; kwargs...)
    update_tf!(annealing, tf)
    k = merge(kwargs, annealing.kwargs)
    solve(annealing.ode_problem, method; k...)
end

function update_tf!(annealing::AbstractAnnealing, tf)
    annealing.ode_problem.p.tf = tf
end

include("piecewise_control.jl")
include("annealing_factory.jl")
