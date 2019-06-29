mutable struct DataStateVector{T} <: DEDataVector{T}
    x::Array{T, 1}
    control
end

mutable struct DataDensityMatrix{T} <: DEDataVector{T}
    x::Array{T, 1}
    control::AnnealingControl
end

mutable struct AdiabaticFramePiecewiseControl <: AnnealingControl
    tf
    adiabatic_scaling
    geometric_scaling
end

function next_state(control::AdiabaticFramePiecewiseControl)
    popfirst!(control.adiabatic_scaling)
    popfirst!(control.geometric_scaling)
end

function annealing_factory(hamiltonian; type="normal")
end

function _piecewise_annealing(H::AdiabaticFrameHamiltonian, sstops, control::AdiabaticFramePiecewiseControl; open_system_update=nothing)
    if open_system_update == nothing
        function f(du, u, p, t)
            hmat = -1.0im * hfun(t)
            mul!(du, hmat, u)
            axpy!(-1.0, u*hmat, du)

        end
    else
    end
end
