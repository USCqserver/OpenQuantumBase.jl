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
    sspan::Tuple
    """A system bath interaction set."""
    interactions::Any
    """Additional control protocols for the annealing."""
    control::Union{AbstractAnnealingControl,Nothing}
    """Extra times that the timestepping algorithm must step to."""
    tstops::Any
end

function Annealing(
    H,
    u0;
    sspan = (0.0, 1.0),
    coupling = nothing,
    bath = nothing,
    control = nothing,
    interactions = nothing,
    tstops = Float64[],
)
    if coupling != nothing && bath != nothing
        if interactions != nothing
            throw(ArgumentError("Both interactions and coupling/bath are specified. Please merge coupling/bath into interactions."))
        end
        interactions = InteractionSet(Interaction(coupling, bath))
    end
    Annealing(H, u0, sspan, interactions, control, tstops)
end

"""
$(TYPEDEF)
Defines a complete set of ODE parameters, which includes Hamiltonian, total annealing time, open system and control objects.
# Fields
$(FIELDS)
"""
struct ODEParams{T<:Union{AbstractFloat,UnitTime}}
    """Hamiltonian"""
    H::Union{AbstractHamiltonian,Nothing}
    """Total annealing time"""
    tf::T
    """Open system object"""
    opensys::Union{AbstractOpenSys,Nothing}
    """Annealing control object"""
    control::Union{AbstractAnnealingControl,Nothing}
end


ODEParams(H, tf::Real; opensys = nothing, control = nothing) =
    ODEParams(H, float(tf), opensys, control)

ODEParams(H, tf::UnitTime; opensys = nothing, control = nothing) =
    ODEParams(H, tf, opensys, control)

ODEParams(tf::Real; opensys = nothing, control = nothing) =
    ODEParams(nothing, float(tf), opensys, control)

ODEParams(tf::UnitTime; opensys = nothing, control = nothing) =
    ODEParams(nothing, tf, opensys, control)
