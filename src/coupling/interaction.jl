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
Base.iterate(iters::InteractionSet, state = 1) =
    Base.iterate(iters.interactions, state)

build_redfield(iset::InteractionSet, U, Ta; atol = 1e-8, rtol = 1e-6) =
    [build_redfield(i, U, Ta, atol = atol, rtol = rtol) for i in iset]
build_redfield(i::Interaction, U, Ta; atol = 1e-8, rtol = 1e-6) =
    build_redfield(i.coupling, i.bath, U, Ta, atol = atol, rtol = rtol)

build_CGG(iset::InteractionSet, U, tf; atol = 1e-8, rtol = 1e-6, Ta = nothing) =
    [build_CGG(i, U, tf, atol = atol, rtol = rtol, Ta = Ta) for i in iset]
build_CGG(i::Interaction, U, tf; atol = 1e-8, rtol = 1e-6, Ta = nothing) =
    build_CGG(i.coupling, i.bath, U, tf, atol = atol, rtol = rtol, Ta = Ta)

build_davies(iset::InteractionSet, ω_range, lambshift::Bool) =
    [build_davies(i, ω_range, lambshift) for i in iset]
build_davies(inter::Interaction, ω_hint, lambshift) =
    build_davies(inter.coupling, inter.bath, ω_hint, lambshift)

build_fluctuator(iset::InteractionSet) =
    [build_fluctuator(i.coupling, i.bath) for i in iset]
build_fluctuator(inter::Interaction) =
    build_fluctuator(inter.coupling, inter.bath)