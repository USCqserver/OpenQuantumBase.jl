"""
$(TYPEDEF)

An object to hold system operator and the corresponding bath object.

$(FIELDS)
"""
struct Interaction
    """system operator"""
    coupling::AbstractCouplings
    """bath coupling to the system operator"""
    bath::AbstractBath
end

"""
$(TYPEDEF)

An container for different system-bath interactions.

$(FIELDS)
"""
struct InteractionSet{T<:Tuple}
    """A tuple of Interaction"""
    interactions::T
end

InteractionSet(inters::Interaction...) = InteractionSet(inters)
Base.length(inters::InteractionSet) = Base.length(inters.interactions)
Base.getindex(inters::InteractionSet, key...) =
    Base.getindex(inters.interactions, key...)

function build_redfield(iset::InteractionSet, U, tf, Ta,; atol = 1e-8, rtol = 1e-6)
    if length(iset) == 1
        build_redfield(iset[1], U, tf, Ta, atol = atol, rtol = rtol)
    else
        RedfieldSet([
            build_redfield(i, U, tf, Ta, atol = atol, rtol = rtol)
            for i in iset.interactions
        ]...)
    end
end

build_redfield(i::Interaction, U, tf, Ta; atol = 1e-8, rtol = 1e-6) =
    build_redfield(i.coupling, i.bath, U, tf, Ta, atol = atol, rtol = rtol)

function build_CGME(
    iset::InteractionSet,
    U,
    tf;
    atol = 1e-8,
    rtol = 1e-6,
    Ta = nothing,
)
    if length(iset) == 1
        build_CGME(iset[1], U, tf, atol = atol, rtol = rtol, Ta = Ta)
    else
        throw(ArgumentError("InteractionSet is not supported for CGME for now."))
    end
end

build_CGME(i::Interaction, U, tf; atol = 1e-8, rtol = 1e-6, Ta = nothing) =
    build_CGME(i.coupling, U, tf, i.bath, atol = atol, rtol = rtol, Ta = Ta)

function build_davies(iset::InteractionSet, ω_range, lambshift::Bool)
    if length(iset) == 1
        build_davies(iset[1], ω_range, lambshift)
    else
        throw(ArgumentError("Multiple interactions is not yet supported for adiabatic master equation solver."))
    end
end

build_davies(inter::Interaction, ω_hint, lambshift) =
    build_davies(inter.coupling, inter.bath, ω_hint, lambshift)
