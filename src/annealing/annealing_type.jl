"""
$(TYPEDEF)
`Annealing` type defines the evolution of a time-dependent Hamiltonian in both closed-system and open-system settings. It is called `Annealing` because HOQST started as a toolbox for quantum annealing.

# Fields
$(FIELDS)
"""
mutable struct Annealing{hType,uType} <: AbstractAnnealing{hType,uType}
    """Hamiltonian for the annealing."""
    H::hType
    """Initial state for the annealing."""
    u0::uType
    """Function of annealing parameter s wrt to t"""
    annealing_parameter::Any
    """A system bath interaction set."""
    interactions::Union{InteractionSet,Nothing}
end

Evolution = Annealing

function Annealing(
    H::AbstractHamiltonian{T},
    u0;
    coupling = nothing,
    bath = nothing,
    interactions = nothing,
    annealing_parameter = (tf, t) -> t / tf,
) where {T}
    if coupling != nothing && bath != nothing
        if interactions != nothing
            throw(ArgumentError("Both interactions and coupling/bath are specified. Please merge coupling/bath into interactions."))
        end
        interactions = InteractionSet(Interaction(coupling, bath))
    end
    if !(T<:Complex)
    end
    Annealing(H, u0, annealing_parameter, interactions)
end

set_u0!(A::Annealing, u0) = A.u0 = u0
set_annealing_parameter!(A::Annealing, param) = A.annealing_parameter = param

"""
$(TYPEDEF)
Defines a complete set of ODE parameters, which includes Hamiltonian, total annealing time, open system and control objects.
# Fields
$(FIELDS)
"""
struct ODEParams
    # H and opensys may be move to dedicated OpenSysOp in the future
    """Hamiltonian"""
    L::Any
    """Total annealing time"""
    tf::Real
    """Function to convert physical time to annealing parameter"""
    annealing_parameter::Function
    """Annealing control object"""
    control::Any
end

ODEParams(L, tf::Real, annealing_param; control = nothing) =
    ODEParams(L, tf, annealing_param, control)
(P::ODEParams)(t::Real) = P.annealing_parameter(P.tf, t)
