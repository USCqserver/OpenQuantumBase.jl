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

function coarse_grain_timescale(
    cfun,
    lim;
    rtol = sqrt(eps()),
    atol = 0,
)
    τsb, err_sb = τ_SB(cfun, rtol = rtol, atol = atol)
    τb, err_b = τ_B(cfun, lim, τsb, rtol = rtol, atol = atol)
    sqrt(τsb * τb / 5)
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
    sqrt(τsb * τb / 5)
end
