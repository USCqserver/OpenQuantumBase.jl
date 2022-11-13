"""
$(TYPEDEF)

Base for types defining stochastic bath object.
"""
abstract type StochasticBath <: AbstractBath end

build_correlation(bath::AbstractBath) =
    SingleFunctionMatrix((t₁, t₂) -> correlation(t₁ - t₂, bath))
build_spectrum(bath::AbstractBath) = (ω) -> spectrum(ω, bath)

"""
$(SIGNATURES)

Calculate the Lamb shift of `bath`. All the keyword arguments of `quadgk` function is supported.
"""
S(w, bath::AbstractBath; kwargs...) = lambshift(w, (ω) -> spectrum(ω, bath); kwargs...)

"""
$(SIGNATURES)

Calculate spectral density ``γ(ω)`` of `bath`.
"""
γ(ω, bath::AbstractBath) = spectrum(ω, bath)

function build_lambshift(ω_range::AbstractVector, turn_on::Bool, bath::AbstractBath, lambshift_kwargs::Dict)
    if turn_on == true
        if isempty(ω_range)
            S_loc = (ω) -> S(ω, bath; lambshift_kwargs...)
        else
            s_list = [S(ω, bath; lambshift_kwargs...) for ω in ω_range]
            S_loc = construct_interpolations(ω_range, s_list)
        end
    else
        S_loc = (ω) -> 0.0
    end
    S_loc
end

# This function is for compatibility purpose
build_lambshift(ω_range::AbstractVector, turn_on::Bool, bath::AbstractBath, ::Nothing) = build_lambshift(ω_range, turn_on, bath, Dict())

"""
$(SIGNATURES)

Calculate the two point correlation function ``C(t1, t2)`` of `bath`. Fall back to `correlation(t1-t2, bath)` unless otherwise defined.
"""
@inline correlation(t1, t2, bath::AbstractBath) = correlation(t1 - t2, bath)

"""
$(SIGNATURES)

Calculate the bath time scale ``τ_{SB}`` from the bath correlation function `cfun`. ``τ_{SB}`` is defined as ``1/τ_{SB}=∫_0^∞ |C(τ)|dτ``. The upper limit of the integration can be modified using keyword `lim`. `atol` and `rtol` are the absolute and relative error tolerance of the integration.
"""
function τ_SB(cfun; lim=Inf, rtol=sqrt(eps()), atol=0)
    res, err = quadgk((x) -> abs(cfun(x)), 0, lim, rtol=rtol, atol=atol)
    1 / res, err / abs2(res)
end

"""
$(SIGNATURES)

Calculate the bath time scale ``τ_{SB}`` for the `bath` object.
"""
τ_SB(bath::AbstractBath; lim=Inf, rtol=sqrt(eps()), atol=0) =
    τ_SB((t) -> correlation(t, bath), lim=lim, rtol=rtol, atol=atol)

"""
$(SIGNATURES)

Calculate the bath time scale ``τ_B`` from the correlation function `cfun`. ``τ_B`` is defined as ``τ_B/τ_{SB} = ∫_0^T τ|C(τ)|dτ``. The upper limit of the integration ``T`` is specified by `lim`. `τsb` is the other bath time scale ``1/τ_{SB}=∫_0^∞ |C(τ)|dτ``. `atol` and `rtol` are the absolute and relative error tolerance for the integration.
"""
function τ_B(cfun, lim, τsb; rtol=sqrt(eps()), atol=0)
    res, err = quadgk((x) -> x * abs(cfun(x)), 0, lim, rtol=rtol, atol=atol)
    res * τsb, err * τsb
end

"""
$(SIGNATURES)

Calculate the bath time scale ``τ_B`` for the `bath` object.
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

Calculate the optimal coarse grain time scale ``T_a=√{τ_{SB}τ_B/5}`` for a total evolution time `lim`. `atol` and `rtol` are the absolute and relative error for the integration.
"""
function coarse_grain_timescale(
    bath::AbstractBath,
    lim;
    rtol=sqrt(eps()),
    atol=0
)
    τsb, err_sb = τ_SB(bath, rtol=rtol, atol=atol)
    τb, err_b = τ_B(bath, lim, τsb, rtol=rtol, atol=atol)
    sqrt(τsb * τb / 5), (err_sb * τb + τsb * err_b) / 10 / sqrt(τsb * τb / 5)
end