"""
$(TYPEDEF)
Defines a quantum annealing process.
# Fields
$(FIELDS)
"""
struct Annealing{hType, uType} <: AbstractAnnealing{hType, uType}
    """Hamiltonian for the annealing."""
    H::hType
    """Initial state for the annealing."""
    u0::uType
    """Range of annealing parameter."""
    sspan
    """A list of system bath coupling operators(system part)."""
    coupling
    """A list of system bath coupling operators(bath part)."""
    bath
    """Additional control protocols for the annealing."""
    control
    """Extra times that the timestepping algorithm must step to."""
    tstops
end


function Annealing(H, u0; sspan=(0.0, 1.0), coupling=nothing, bath=nothing, control=nothing, tstops=[])
    Annealing(H, u0, sspan, coupling, bath, control, tstops)
end

function solve_unitary(A::Annealing, tf::Real; kwargs...)
    u0 = Matrix{ComplexF64}(I, size(A.H(0.0)))
    p = AnnealingParams(A.H, float(tf))
    ff = ODEFunction(mul_ode; jac = mul_jac)
    prob = ODEProblem(ff, u0, A.sspan, p)
    # currently stiff algorithm does not support complex type
    solve(prob; alg_hints = [:nonstiff], tstops=A.tstops, kwargs...)
end

function solve_schrodinger(A::Annealing, tf::Real; kwargs...)
    if ndims(A.u0) != 1
        throw(DomainError("Initial state must be given as a state vector."))
    end
    u0 = A.u0
    p = AnnealingParams(A.H, float(tf))
    ff = ODEFunction(mul_ode; jac = mul_jac)
    prob = ODEProblem(ff, u0, A.sspan, p)
    solve(prob; alg_hints = [:nonstiff], tstops=A.tstops, kwargs...)
end

function solve_von_neumann(A::Annealing, tf::Real; kwargs...)
    if ndims(A.u0) == 1
        u0 = A.u0*A.u0'
    else
        u0 = A.u0
    end
    p = AnnealingParams(A.H, float(tf))
    prob = ODEProblem(von_neumann_ode, u0, A.sspan, p)
    solve(prob; alg_hints = [:nonstiff], tstops=A.tstops, kwargs...)
end

function solve_redfield(A::Annealing, tf::Real, unitary; kwargs...)
    if ndims(A.u0) == 1
        u0 = A.u0*A.u0'
    else
        u0 = A.u0
    end
    opensys = create_redfield(A.coupling, unitary, tf, A.bath)
    p = AnnealingParams(A.H, float(tf); opensys=opensys)
    prob = ODEProblem(von_neumann_open_ode, u0, A.sspan, p)
    solve(prob; alg_hints = [:nonstiff], tstops=A.tstops, kwargs...)
end

function mul_ode(du, u, p, t)
    mul!(du, p.H(t), u)
    lmul!(-1.0im * p.tf, du)
end

function mul_jac(J, u, p, t)
    hmat = p.H(t)
    mul!(J, -1.0im * p.tf, hmat)
end

function von_neumann_ode(du, u, p, t)
    fill!(du, 0.0+0.0im)
    p.H(du, u, p.tf, t)
end

function von_neumann_open_ode(du, u, p, t)
    fill!(du, 0.0+0.0im)
    p.H(du, u, p.tf, t)
    p.opensys(du, u, p, t)
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
