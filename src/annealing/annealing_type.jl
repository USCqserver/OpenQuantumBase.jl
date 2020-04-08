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
    coupling::Union{AbstractCouplings, Nothing}
    """A list of system bath coupling operators(bath part)."""
    bath
    """A system bath interaction set."""
    interactions
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
    interactions = nothing,
    tstops = Float64[]
)
    if interactions == nothing
        Annealing(H, u0, sspan, coupling, bath, nothing, control, tstops)
    else
        Annealing(H, u0, sspan, nothing, nothing, interactions, control, tstops)
    end
end
