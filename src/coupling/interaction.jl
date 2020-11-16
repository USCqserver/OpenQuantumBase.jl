"""
$(TYPEDEF)

Base for types defining system bath interactions in open quantum system models.
"""
abstract type AbstractInteraction end

"""
$(TYPEDEF)

An object to hold coupling operator and the corresponding bath object.

$(FIELDS)
"""
struct Interaction <: AbstractInteraction
    """system operator"""
    coupling::AbstractCouplings
    """bath coupling to the system operator"""
    bath::AbstractBath
end

"""
$(TYPEDEF)

A Lindblad operator, define by a rate ``γ`` and corresponding operator ``L```.

$(FIELDS)
"""
struct Lindblad <: AbstractInteraction
    """Lindblad rate"""
    γ::Any
    """Lindblad operator"""
    L::Any
    """size"""
    size::Tuple
end

Lindblad(γ::Number, L::Matrix) = Lindblad((s) -> γ, (s) -> L, size(L))
Lindblad(γ::Number, L) = Lindblad((s) -> γ, L, size(L(0)))
Lindblad(γ, L::Matrix) = Lindblad(γ, (s) -> L, size(L))

function Lindblad(γ, L)
    if !(typeof(γ(0)) <: Number)
        throw(ArgumentError("γ should return a number."))
    end
    if !(typeof(L(0)) <: Matrix)
        throw(ArgumentError("L should return a matrix."))
    end
    Lindblad(γ, L, size(L(0)))
end

Base.size(lind::Lindblad) = lind.size

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

InteractionSet(inters::AbstractInteraction...) = InteractionSet(inters)
Base.length(inters::InteractionSet) = Base.length(inters.interactions)
Base.getindex(inters::InteractionSet, key...) =
    Base.getindex(inters.interactions, key...)
Base.iterate(iters::InteractionSet, state=1) =
    Base.iterate(iters.interactions, state)

# the following functions are used to build different Liouvillians
# from the InteractionSet.
function redfield_from_interactions(iset::InteractionSet, U, Ta, atol, rtol)
    kernels = [build_redfield_kernel(i) for i in iset]
    RedfieldLiouvillian(kernels, U, Ta, atol, rtol)
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

build_lindblad(iset::InteractionSet) = 
    LindbladLiouvillian([i for i in iset if typeof(i)<:Lindblad])