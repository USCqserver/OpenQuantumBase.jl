"""
$(TYPEDEF)
`Annealing` type defines the evolution of a time-dependent Hamiltonian in both closed-system and open-system settings. It is called `Annealing` because HOQST started as a toolbox for quantum annealing.

# Fields
$(FIELDS)
"""
mutable struct Annealing{constant_hamiltonian} <: AbstractAnnealing{constant_hamiltonian}
    "Hamiltonian for the annealing."
    H
    "Initial state for the annealing."
    u0
    "Function of annealing parameter s wrt to t"
    annealing_parameter::Any
    "A system bath interaction set."
    interactions::Union{InteractionSet,Nothing}
end

Evolution = Annealing

function Annealing(
    H::AbstractHamiltonian{T},
    u0;
    coupling=nothing,
    bath=nothing,
    interactions=nothing,
    annealing_parameter=(tf, t) -> t / tf
) where {T}
    if !isnothing(coupling) && !isnothing(bath)
        if !isnothing(interactions)
            throw(ArgumentError("Both interactions and coupling/bath are specified. Please merge coupling/bath into interactions."))
        end
        interactions = InteractionSet(Interaction(coupling, bath))
    end
    H = (T <: Complex) ? H : convert(Complex, H)
    Annealing{isconstant(H)}(H, u0, annealing_parameter, interactions)
end

set_u0!(A::Annealing, u0) = A.u0 = u0
set_annealing_parameter!(A::Annealing, param) = A.annealing_parameter = param

"""
$(TYPEDEF)

The `ODEParams` struct represents a complete set of parameters for an Ordinary 
Differential Equation (ODE), including the Liouville operator, total evolution 
time, an annealing parameter function, and an object storing accepted keyword 
arguments for subroutines.

# Fields
$(FIELDS)
"""
struct ODEParams
    "Liouville operator"
    L::Any
    "Total evolution time"
    tf::Real
    "Function to convert physical time to annealing parameter"
    annealing_parameter::Function
    "Keyword arguments for subroutines"
    accepted_kwargs::Any
end

# TODO: filter the keyword argument
ODEParams(L, tf::Real, annealing_param; kwargs...) =
    ODEParams(L, tf, annealing_param, kwargs)
(P::ODEParams)(t::Real) = P.annealing_parameter(P.tf, t)
