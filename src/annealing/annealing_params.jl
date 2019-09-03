"""
$(TYPEDEF)

Base for types defining [parameters](http://docs.juliadiffeq.org/latest/tutorials/ode_example.html) for annealing ODEs.
"""
abstract type AbstractAnnealingParams end


"""
$(TYPEDEF)
Defines a complete set of parameters, which includes Hamiltonian, total annealing time, open system and control objects.
# Fields
$(FIELDS)
"""
struct AnnealingParams{T<:Union{
    AbstractFloat,
    UnitTime
}} <: AbstractAnnealingParams
    """Hamiltonian"""
    H::AbstractHamiltonian
    """Total annealing time"""
    tf::T
    """Open system object"""
    opensys
    """Annealing control object"""
    control
end

function set_tf(P::AnnealingParams{T}, tf::Real) where T <: AbstractFloat
    AnnealingParams(P.H, float(tf), P.opensys, P.control)
end

function set_tf(P::AnnealingParams{T}, tf::Real) where T <: UnitTime
    AnnealingParams(P.H, UnitTime(tf), P.opensys, P.control)
end

function AnnealingParams(
    H,
    tf::T;
    opensys = nothing, control = nothing
) where T <: Number
    AnnealingParams(H, float(tf), opensys, control)
end

function AnnealingParams(H, tf::UnitTime; opensys = nothing, control = nothing)
    AnnealingParams(H, tf, opensys, control)
end


"""
$(TYPEDEF)
Defines a light version of [parameters](http://docs.juliadiffeq.org/latest/tutorials/ode_example.html) for annealing ODEs.
# Fields
$(FIELDS)
"""
struct LightAnnealingParams{T<:Union{
    AbstractFloat,
    UnitTime
}} <: AbstractAnnealingParams
    """Total annealing time"""
    tf::T
    """Annealing control object"""
    control
end

function LightAnnealingParams(tf::T; control = nothing) where T <: Number
    LightAnnealingParams(float(tf), control)
end

function LightAnnealingParams(tf::UnitTime; control = nothing)
    LightAnnealingParams(tf, control)
end

function set_tf(P::LightAnnealingParams{T}, tf::Real) where T <: AbstractFloat
    LightAnnealingParams(float(tf), P.control)
end

function set_tf(P::LightAnnealingParams{T}, tf::Real) where T <: UnitTime
    LightAnnealingParams(UnitTime(tf), P.control)
end
