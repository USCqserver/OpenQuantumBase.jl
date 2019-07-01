struct Annealing <: AbstractAnnealing
    ode_problem
    kwargs
end

function update_tf!(annealing::AbstractAnnealing, tf)
    annealing.ode_problem.p.tf = tf
end

include("piecewise_control.jl")
include("annealing_factory.jl")
