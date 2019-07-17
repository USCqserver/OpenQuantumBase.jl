function annealing_factory(H::AbstractHamiltonian, u0, tf; kwargs...)
    u_dim = ndims(u0)
    if haskey(kwargs, :output_type)
        output_type = kwargs[:output_type]
    else
        output_type = nothing
    end

    if isnothing(output_type)

    else
    end

    if output_type == "unitary"
        u0 = Matrix{ComplexF64}(I, size(H(0.0)))
    elseif output_type == "state_vector"
        u0 =
    elseif output_type == "density_matrix"

    end

end

function annealing_factory(H::AdiabaticFrameHamiltonian, u0)
    if ndims(u0) == 1
        ρ0 = u0*u0'
    end
    function f(du, u, p, t)
        fill!(du, 0.0+0.0im)
        H(du, u, p, t)
    end
    prob = ODEProblem(f, ρ0, (0.0, 1.0), 1.0)
end


function annealing_factory(H::AdiabaticFrameHamiltonian, u0, sp, sd; additional_stops=[])
    control = pausing_annealing_parameter(sp, sd)
    if ndims(u0) == 1
        u0 = u0*u0'
    end
    ρ0 = ControlDensityMatrix(u0, 1)
    # create stops
    sstops = vcat(control.stops, additional_stops)
    # create callback
    function condition(u,t,integrator)
        t in integrator.p.stops
    end
    function affect!(integrator)
        for c in full_cache(integrator)
            c.stage += 1
        end
        # the state of the system does not change
        u_modified!(integrator, false)
    end
    cb = DiscreteCallback(condition, affect!)
    # create control
    function f(du, u, p, t)
        fill!(du, 0.0+0.0im)
        H(du, u, p, t)
    end
    prob = ODEProblem(f, ρ0, (0.0, control.sf), control)
    Annealing(prob, Dict(:callback=>cb, :tstops=>sstops))
end
