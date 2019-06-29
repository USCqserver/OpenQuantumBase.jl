mutable struct DataStateVector{T} <: DEDataVector{T}
    x::Array{T, 1}
    control
end

mutable struct DataDensityMatrix{T} <: DEDataVector{T}
    x::Array{T, 2}
    control
end

mutable struct AdiabaticFramePiecewiseControl <: AnnealingControl
    tf::Float64
    annealing_parameter
    geometric_scaling
end

function next_state(control::AdiabaticFramePiecewiseControl)
    popfirst!(control.annealing_parameter)
    popfirst!(control.geometric_scaling)
end

function annealing_factory(hamiltonian; type="normal")
end

function _piecewise_annealing(H::AdiabaticFrameHamiltonian, u0, sstops, control::AdiabaticFramePiecewiseControl; open_system_update=nothing)
    if open_system_update == nothing
        # callback function
        function condition(u,t,integrator)
            t in sstops
        end
        function affect!(integrator)
            next_state(integrator.p.control)
            for c in full_cache(integrator)
                next_state(c.control)
            end
        end
        prob = ODEProblem(H, u0, (0.0,1.0))
    else
    end
end
