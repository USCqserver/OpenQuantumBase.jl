struct Annealing{hType, uType} <: AbstractAnnealing{hType, uType}
    H::hType
    u0::uType
end

function calculate_unitary(A::Annealing, tf::Real; alg=nothing, kwargs...)
    u0 = Matrix{ComplexF64}(I, size(A.H(0.0)))
    p = AnnealingParams(H, float(tf))
    ff = ODEFunction(unitary_odef; jac = unitary_jac)
    prob = ODEProblem(ff, u0, (0.0, 1.0), tf)
    # currently stiff algorithm does not support complex type
    if alg==nothing
        solve(prob, alg_hints = [:nonstiff]; kwargs...)
    else
        solve(prob, alg; kwargs...)
    end
end

function unitary_odef(du, u, p, t)
    mul!(du, p.H(t), u)
    lmul!(-1.0im * p.tf, du)
end

function unitary_jac(J, u, p, t)
    hmat = p.H(t)
    mul!(J, -1.0im * p.tf, hmat)
end

# function solve(annealing::Annealing, method, tf::Number; kwargs...)
#     update_tf!(annealing, tf)
#     k = merge(kwargs, annealing.kwargs)
#     solve(annealing.ode_problem, method; k...)
# end
#
# function update_tf!(annealing::AbstractAnnealing, tf)
#     annealing.ode_problem.p.tf = tf
# end
