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
    "system operator"
    coupling::AbstractCouplings
    "bath coupling to the system operator"
    bath::AbstractBath
end

isconstant(x::Interaction) = isconstant(x.coupling)
rotate(i::Interaction, v) = Interaction(rotate(i.coupling, v), i.bath)

"""
$(TYPEDEF)

A Lindblad operator, define by a rate ``γ`` and corresponding operator ``L```.

$(FIELDS)
"""
struct Lindblad <: AbstractInteraction
    "Lindblad rate"
    γ::Any
    "Lindblad operator"
    L::Any
    "size"
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

"""
$(TYPEDEF)

An container for different system-bath interactions.

$(FIELDS)
"""
struct InteractionSet{T<:Tuple}
    "A tuple of Interaction"
    interactions::T
end

InteractionSet(inters::AbstractInteraction...) = InteractionSet(inters)
rotate(inters::InteractionSet, v) = InteractionSet([rotate(i, v) for i in inters]...)
Base.length(inters::InteractionSet) = Base.length(inters.interactions)
Base.getindex(inters::InteractionSet, key...) =
    Base.getindex(inters.interactions, key...)
Base.iterate(iters::InteractionSet, state=1) =
    Base.iterate(iters.interactions, state)
# The following functions are used to build different Liouvillians from
# the `InteractionSet`. They are not publicly available in the current
# release.
function redfield_from_interactions(iset::InteractionSet, U, Ta, atol, rtol)
    kernels = [build_redfield_kernel(i) for i in iset if !(typeof(i.bath) <: StochasticBath)]
    [RedfieldLiouvillian(kernels, U, Ta, atol, rtol)]
end

function cg_from_interactions(iset::InteractionSet, U, tf, Ta, atol, rtol)
    if Ta === nothing || ndims(Ta) == 0
        kernels = [build_cg_kernel(i, tf, Ta) for i in iset if !(typeof(i.bath) <: StochasticBath)]
    else
        if length(Ta) != length(iset)
            throw(ArgumentError("Ta should have the same length as the interaction sets."))
        end
        kernels = [build_cg_kernel(i, tf, t) for (i, t) in zip(iset, Ta) if !(typeof(i.bath) <: StochasticBath)]
    end
    [CGLiouvillian(kernels, U, atol, rtol)]
end

function ule_from_interactions(iset::InteractionSet, U, Ta, atol, rtol)
    kernels = [build_ule_kernel(i) for i in iset if !(typeof(i.bath) <: StochasticBath)]
    [ULELiouvillian(kernels, U, Ta, atol, rtol)]
end

function davies_from_interactions(iset::InteractionSet, ω_range, lambshift::Bool, lambshift_kwargs)
    davies_list = []
    for i in iset
        coupling = i.coupling
        bath = i.bath
        if !(typeof(bath) <: StochasticBath)
            γfun = build_spectrum(bath)
            Sfun = build_lambshift(ω_range, lambshift, bath, lambshift_kwargs)
            if typeof(bath) <: CorrelatedBath
                push!(davies_list, CorrelatedDaviesGenerator(coupling, γfun, Sfun, build_inds(bath)))
            else
                push!(davies_list, DaviesGenerator(coupling, γfun, Sfun))
            end
        end
    end
    davies_list
end

function davies_from_interactions(gap_idx, iset::InteractionSet, ω_range, lambshift::Bool, lambshift_kwargs::Dict)
    davies_list = []
    for i in iset
        coupling = i.coupling
        bath = i.bath
        if !(typeof(bath) <: StochasticBath)
            γfun = build_spectrum(bath)
            Sfun = build_lambshift(ω_range, lambshift, bath, lambshift_kwargs)
            if typeof(bath) <: CorrelatedBath
                # TODO: optimize the performance of `CorrelatedDaviesGenerator`` if `coupling` is constant
                push!(davies_list, build_const_correlated_davies(coupling, gap_idx, γfun, Sfun, build_inds(bath)))
            else
                push!(davies_list, build_const_davies(coupling, gap_idx, γfun, Sfun))
            end
        end
    end
    davies_list
end

function onesided_ame_from_interactions(iset::InteractionSet, ω_range, lambshift::Bool, lambshift_kwargs)
    l_list = []
    for i in iset
        coupling = i.coupling
        bath = i.bath
        if !(typeof(bath) <: StochasticBath)
            γfun = build_spectrum(bath)
            Sfun = build_lambshift(ω_range, lambshift, bath, lambshift_kwargs)
            if typeof(i.bath) <: CorrelatedBath
                inds = build_inds(bath)
            else
                inds = ((i, i) for i in 1:length(coupling))
                γfun = SingleFunctionMatrix(γfun)
                Sfun = SingleFunctionMatrix(Sfun)
            end
            push!(l_list, OneSidedAMELiouvillian(coupling, γfun, Sfun, inds))
        end
    end
    l_list
end

function fluctuator_from_interactions(iset::InteractionSet)
    f_list = []
    for i in iset
        coupling = i.coupling
        bath = i.bath
        if typeof(bath) <: EnsembleFluctuator
            num = length(coupling)
            dist = construct_distribution(bath)
            b0 = [x.b for x in bath.f] .* rand([-1, 1], length(dist), num)
            next_τ, next_idx = findmin(rand(dist, num))
            push!(f_list, FluctuatorLiouvillian(coupling, dist, b0, next_idx, next_τ, sum(b0, dims=1)[:]))
        end
    end
    f_list
end

lindblad_from_interactions(iset::InteractionSet) =
    [LindbladLiouvillian([i for i in iset if typeof(i) <: Lindblad])]

function build_redfield_kernel(i::Interaction)
    coupling = i.coupling
    bath = i.bath
    cfun = build_correlation(bath)
    rinds = typeof(cfun) == SingleFunctionMatrix ?
            ((i, i) for i = 1:length(coupling)) : build_inds(bath)
    # the kernels is current set as a tuple
    (rinds, coupling, cfun)
end

function build_cg_kernel(i::Interaction, tf, Ta)
    coupling = i.coupling
    cfun = build_correlation(i.bath)
    Ta = Ta === nothing ? coarse_grain_timescale(i.bath, tf)[1] : Ta
    rinds = typeof(cfun) == SingleFunctionMatrix ?
            ((i, i) for i = 1:length(coupling)) : build_inds(i.bath)
    # the kernels is current set as a tuple
    (rinds, coupling, cfun, Ta)
end

function build_ule_kernel(i::Interaction)
    coupling = i.coupling
    cfun = build_jump_correlation(i.bath)
    rinds = typeof(cfun) == SingleFunctionMatrix ?
            ((i, i) for i = 1:length(coupling)) : build_inds(i.bath)
    # the kernels is current set as a tuple
    (rinds, coupling, cfun)
end