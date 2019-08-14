function adjust_sspan(control, sspan) end

"""
$(TYPEDEF)
Defines a quantum annealing process.
# Fields
$(FIELDS)
"""
struct Annealing{hType,uType} <: AbstractAnnealing{hType,uType}
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


function Annealing(
    H,
    u0;
    sspan = (0.0, 1.0),
    coupling = nothing,
    bath = nothing,
    control = nothing,
    tstops = []
)
    if control!=nothing
        sspan = adjust_sspan(control, sspan)
    end
    Annealing(H, u0, sspan, coupling, bath, control, tstops)
end
