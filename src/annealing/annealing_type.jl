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
    """A list of system bath coupling operators(system part)."""
    coupling::Union{AbstractCouplings,Nothing}
    """A list of system bath coupling operators(bath part)."""
    bath::Union{AbstractBath,Nothing}
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
    if interactions == nothing
        Annealing(H, u0, sspan, coupling, bath, nothing, control, tstops)
    else
        Annealing(H, u0, sspan, nothing, nothing, interactions, control, tstops)
    end
end


"""
$(TYPEDEF)
Defines a complete set of ODE parameters, which includes Hamiltonian, total annealing time, open system and control objects.
# Fields
$(FIELDS)
"""
struct ODEParams{T<:Union{AbstractFloat,UnitTime}}
    """Hamiltonian"""
    H::AbstractHamiltonian
    """Total annealing time"""
    tf::T
    """Open system object"""
    opensys::Union{AbstractOpenSys,Nothing}
    """Annealing control object"""
    control::Union{AbstractAnnealingControl,Nothing}
end


function ODEParams(
    H,
    tf::T;
    opensys = nothing,
    control = nothing,
) where {T<:Number}
    ODEParams(H, float(tf), opensys, control)
end

ODEParams(H, tf::UnitTime; opensys = nothing, control = nothing) =
    ODEParams(H, tf, opensys, control)

ODEParams(tf; opensys = nothing, control = nothing) =
    ODEParams(nothing, tf, opensys, control)
