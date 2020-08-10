build_correlation(bath::AbstractBath) = (τ) -> correlation(τ, bath)
build_spectrum(bath::AbstractBath) = (ω) -> spectrum(ω, bath)

"""
$(SIGNATURES)

Calculate ``τ_{SB}`` from the bath correlation function `cfun`. It is defined as the integration of the absolute value of bath correlation function from zero to `lim`. The default value of `lim` is `Inf`. `atol` and `rtol` are the absolute and relative error of the integration.
"""
function τ_SB(cfun; lim = Inf, rtol = sqrt(eps()), atol = 0)
    res, err = quadgk((x) -> abs(cfun(x)), 0, lim, rtol = rtol, atol = atol)
    1 / res, err / abs2(res)
end

τ_SB(bath::AbstractBath; lim = Inf, rtol = sqrt(eps()), atol = 0) =
    τ_SB((t) -> correlation(t, bath), lim = lim, rtol = rtol, atol = atol)

"""
$(SIGNATURES)

Calculate the bath correlation time ``τ_B``. The upper limit `lim` and `τsb` need to be manually specified. `atol` and `rtol` are the absolute and relative error for the integration.
"""
function τ_B(cfun, lim, τsb; rtol = sqrt(eps()), atol = 0)
    res, err = quadgk((x) -> x * abs(cfun(x)), 0, lim, rtol = rtol, atol = atol)
    res * τsb, err * τsb
end

τ_B(bath::AbstractBath, lim, τsb; rtol = sqrt(eps()), atol = 0) =
    τ_B((t) -> correlation(t, bath), lim, τsb; rtol = rtol, atol = atol)

function coarse_grain_timescale(cfun, lim; rtol = sqrt(eps()), atol = 0)
    τsb, err_sb = τ_SB(cfun, rtol = rtol, atol = atol)
    τb, err_b = τ_B(cfun, lim, τsb, rtol = rtol, atol = atol)
    sqrt(τsb * τb / 5), (err_sb * τb + τsb * err_b) / 10 / sqrt(τsb * τb / 5)
end

"""
$(SIGNATURES)

Calculate the optimal coarse grain time scale ``T_a`` for a total evolution time `lim`. `atol` and `rtol` are the absolute and relative error for the integration.
"""
function coarse_grain_timescale(
    bath::AbstractBath,
    lim;
    rtol = sqrt(eps()),
    atol = 0,
)
    τsb, err_sb = τ_SB(bath, rtol = rtol, atol = atol)
    τb, err_b = τ_B(bath, lim, τsb, rtol = rtol, atol = atol)
    sqrt(τsb * τb / 5), (err_sb * τb + τsb * err_b) / 10 / sqrt(τsb * τb / 5)
end

function build_redfield(
    coupling::AbstractCouplings,
    bath::AbstractBath,
    unitary,
    Ta;
    atol = 1e-8,
    rtol = 1e-6,
)
    cfun = build_correlation(bath)
    DiagRedfieldGenerator(coupling, unitary, cfun, Ta, atol = atol, rtol = rtol)
end

function build_davies(
    coupling::AbstractCouplings,
    bath::AbstractBath,
    ω_range::AbstractVector,
    lambshift::Bool,
)
    if lambshift == true
        if isempty(ω_range)
            S_loc = (ω) -> S(ω, bath)
        else
            s_list = [S(ω, bath) for ω in ω_range]
            S_loc = construct_interpolations(ω_range, s_list)
        end
    else
        S_loc = (ω) -> 0.0
    end
    DaviesGenerator(coupling, build_spectrum(bath), S_loc)
end

function build_CGG(
    coupling::AbstractCouplings,
    bath::AbstractBath,
    unitary,
    tf;
    atol = 1e-8,
    rtol = 1e-6,
    Ta = nothing,
)
    Ta = Ta == nothing ? coarse_grain_timescale(bath, tf)[1] : Ta
    cfun = build_correlation(bath)
    CGGenerator(coupling, unitary, cfun, Ta, atol = atol, rtol = rtol)
end
