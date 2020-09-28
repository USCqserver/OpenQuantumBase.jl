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

function build_redfield_kernel(i::Interaction)
    coupling = i.coupling
    cfun = build_correlation(i.bath)
    rinds = typeof(cfun) == SingleCorrelation ?
        ((i, i) for i = 1:length(coupling)) : build_inds(i.bath)
    # the kernels is current set as a tuple
    (rinds, coupling, cfun)
end

function build_CG_kernel(i::Interaction, tf, Ta)
    coupling = i.coupling
    cfun = build_correlation(i.bath)
    Ta = Ta === nothing ? coarse_grain_timescale(i.bath, tf)[1] : Ta
    rinds = typeof(cfun) == SingleCorrelation ?
        ((i, i) for i = 1:length(coupling)) : build_inds(i.bath)
    # the kernels is current set as a tuple
    (rinds, coupling, cfun, Ta)
end

function build_ule_kernel(i::Interaction)
    coupling = i.coupling
    cfun = build_jump_correlation(i.bath)
    rinds = typeof(cfun) == SingleCorrelation ?
        ((i, i) for i = 1:length(coupling)) : build_inds(i.bath)
    # the kernels is current set as a tuple
    (rinds, coupling, cfun)
end

"""
$(TYPEDEF)

An container for different system-bath interactions.

$(FIELDS)
"""
struct InteractionSet{T <: Tuple}
    """A tuple of Interaction"""
    interactions::T
end

InteractionSet(inters::Interaction...) = InteractionSet(inters)
Base.length(inters::InteractionSet) = Base.length(inters.interactions)
Base.getindex(inters::InteractionSet, key...) =
    Base.getindex(inters.interactions, key...)
Base.iterate(iters::InteractionSet, state=1) =
    Base.iterate(iters.interactions, state)

function build_redfield(iset::InteractionSet, U, Ta, atol, rtol)
    kernels = [build_redfield_kernel(i) for i in iset]
    RedfieldGenerator(kernels, U, Ta, atol, rtol)
end

function build_CGG(iset::InteractionSet, U, tf, Ta, atol, rtol)
    if Ta === nothing || ndims(Ta) == 0
        kernels = [build_CG_kernel(i, tf, Ta) for i in iset]
    else
        if length(Ta) != length(iset)
            throw(ArgumentError("Ta should have the same length as the interaction sets."))
        end
        kernels = [build_CG_kernel(i, tf, t) for (i, t) in zip(iset, Ta)]
    end
    CGGenerator(kernels, U, atol, rtol)
end

function build_ule(iset::InteractionSet, U, Ta, atol, rtol)
    kernels = [build_ule_kernel(i) for i in iset]
    ULindblad(kernels, U, Ta, atol, rtol)
end

build_davies(iset::InteractionSet, ω_range, lambshift::Bool) =
    [build_davies(i, ω_range, lambshift) for i in iset]
build_davies(inter::Interaction, ω_hint, lambshift) =
    build_davies(inter.coupling, inter.bath, ω_hint, lambshift)

build_onesided_ame(iset::InteractionSet, ω_range, lambshift::Bool) =
    [build_onesided_ame(i, ω_range, lambshift) for i in iset]
build_onesided_ame(inter::Interaction, ω_hint, lambshift) =
    build_onesided_ame(inter.coupling, inter.bath, ω_hint, lambshift)

build_fluctuator(iset::InteractionSet) =
    [build_fluctuator(i.coupling, i.bath) for i in iset]
build_fluctuator(inter::Interaction) =
    build_fluctuator(inter.coupling, inter.bath)
