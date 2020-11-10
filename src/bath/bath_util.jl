# this is a quick solution for the single correlation case
struct SingleCorrelation
    cfun::Any
end
@inline Base.getindex(C::SingleCorrelation, ind...) = C.cfun
struct SingleSpectrum
    sfun::Any
end
@inline Base.getindex(S::SingleSpectrum, ind...) = S.sfun
build_correlation(bath::AbstractBath) =
    SingleCorrelation((t₁, t₂) -> correlation(t₁ - t₂, bath))
build_spectrum(bath::AbstractBath) = (ω) -> spectrum(ω, bath)

"""
$(SIGNATURES)

Calculate the Lamb shift of `bath`. `atol` is the absolute tolerance for Cauchy principal value integral.
"""
S(w, bath::AbstractBath; atol=1e-7) = lambshift(w, (ω) -> spectrum(ω, bath), atol=atol)

function build_lambshift(ω_range::AbstractVector, turn_on::Bool, bath::AbstractBath)
    if turn_on == true
        if isempty(ω_range)
            S_loc = (ω) -> S(ω, bath)
        else
            s_list = [S(ω, bath) for ω in ω_range]
            S_loc = construct_interpolations(ω_range, s_list)
        end
    else
        S_loc = (ω) -> 0.0
    end
    S_loc
end

"""
$(SIGNATURES)

Calculate the two point correlation function ``C(t1, t2)`` of `bath`. Fall back to `correlation(t1-t2, bath)` unless otherwise defined.
"""
@inline correlation(t1, t2, bath::AbstractBath) = correlation(t1 - t2, bath)

"""
$(SIGNATURES)

Calculate ``τ_{SB}`` from the bath correlation function `cfun`. It is defined as the integration of the absolute value of bath correlation function from zero to `lim`. The default value of `lim` is `Inf`. `atol` and `rtol` are the absolute and relative error of the integration.
"""
function τ_SB(cfun; lim=Inf, rtol=sqrt(eps()), atol=0)
    res, err = quadgk((x) -> abs(cfun(x)), 0, lim, rtol=rtol, atol=atol)
    1 / res, err / abs2(res)
end

"""
$(SIGNATURES)

Calculate ``τ_{SB}`` from `bath`.
"""
τ_SB(bath::AbstractBath; lim=Inf, rtol=sqrt(eps()), atol=0) =
    τ_SB((t) -> correlation(t, bath), lim=lim, rtol=rtol, atol=atol)

"""
$(SIGNATURES)

Calculate the bath correlation time ``τ_B`` from the bath correlation function `cfun`. The upper limit `lim` and `τsb` need to be manually specified. `atol` and `rtol` are the absolute and relative error for the integration.
"""
function τ_B(cfun, lim, τsb; rtol=sqrt(eps()), atol=0)
    res, err = quadgk((x) -> x * abs(cfun(x)), 0, lim, rtol=rtol, atol=atol)
    res * τsb, err * τsb
end

"""
$(SIGNATURES)

Calculate the bath correlation time ``τ_B`` from `bath`.
"""
τ_B(bath::AbstractBath, lim, τsb; rtol=sqrt(eps()), atol=0) =
    τ_B((t) -> correlation(t, bath), lim, τsb; rtol=rtol, atol=atol)

function coarse_grain_timescale(cfun, lim; rtol=sqrt(eps()), atol=0)
    τsb, err_sb = τ_SB(cfun, rtol=rtol, atol=atol)
    τb, err_b = τ_B(cfun, lim, τsb, rtol=rtol, atol=atol)
    sqrt(τsb * τb / 5), (err_sb * τb + τsb * err_b) / 10 / sqrt(τsb * τb / 5)
end

"""
$(SIGNATURES)

Calculate the optimal coarse grain time scale ``T_a`` for a total evolution time `lim`. `atol` and `rtol` are the absolute and relative error for the integration.
"""
function coarse_grain_timescale(
    bath::AbstractBath,
    lim;
    rtol=sqrt(eps()),
    atol=0,
)
    τsb, err_sb = τ_SB(bath, rtol=rtol, atol=atol)
    τb, err_b = τ_B(bath, lim, τsb, rtol=rtol, atol=atol)
    sqrt(τsb * τb / 5), (err_sb * τb + τsb * err_b) / 10 / sqrt(τsb * τb / 5)
end

function build_davies(
    coupling::AbstractCouplings,
    bath::AbstractBath,
    ω_range::AbstractVector,
    lambshift::Bool,
)
    S_loc = build_lambshift(ω_range, lambshift, bath)
    DaviesGenerator(coupling, build_spectrum(bath), S_loc)
end

function build_onesided_ame(
    coupling::AbstractCouplings,
    bath::AbstractBath,
    ω_range::AbstractVector,
    lambshift::Bool,
)
    gamma = build_spectrum(bath)
    S_loc = build_lambshift(ω_range, lambshift, bath)
    if typeof(bath) <: CorrelatedBath
        inds = build_inds(bath)
    else
        inds = ((i, i) for i in 1:length(coupling))
        gamma = SingleSpectrum(gamma)
        S_loc = SingleSpectrum(S_loc)
    end
    OneSidedAMEGenerator(coupling, gamma, S_loc, inds)
end