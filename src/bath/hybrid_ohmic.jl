"""
$(TYPEDEF)

A hybrid noise model with both low and high frequency noise. The high frequency noise is characterized by Ohmic bath and the low frequence noise is characterized by the MRT width `W`.

$(FIELDS)
"""
struct HybridOhmicBath <: AbstractBath
    "MRT width (2π GHz)"
    W::Float64
    "low spectrum reorganization energy (2π GHz)"
    ϵl::Float64
    "strength of high frequency Ohmic bath"
    η::Float64
    "cutoff frequency"
    ωc::Float64
    "inverse temperature"
    β::Float64
end

function Base.show(io::IO, ::MIME"text/plain", m::HybridOhmicBath)
    print(
        io,
        "Hybrid Ohmic bath instance:\n",
        "W (mK): ",
        freq_2_temperature(m.W / 2 / pi),
        "\n",
        "ϵl (GHz): ",
        m.ϵl / 2 / pi,
        "\n",
        "η (unitless): ",
        m.η / 2 / π,
        "\n",
        "ωc (GHz): ",
        m.ωc / pi / 2,
        "\n",
        "T (mK): ",
        β_2_temperature(m.β)
    )
end

"""
    HybridOhmic(W, η, fc, T)

Construct HybridOhmicBath object with parameters in physical units. `W`: MRT width (mK); `η`: interaction strength (unitless); `fc`: Ohmic cutoff frequency (GHz); `T`: temperature (mK).
"""
function HybridOhmic(W, η, fc, T)
    # scaling of W to the unit of angular frequency
    W = 2 * pi * temperature_2_freq(W)
    # scaling of η because different definition of Ohmic spectrum
    η = 2 * π * η
    β = temperature_2_β(T)
    ωc = 2 * π * fc
    ϵl = W^2 * β / 2
    HybridOhmicBath(W, ϵl, η, ωc, β)
end

"""
    Gₕ(ω, bath::HybridOhmicBath)

High frequency noise spectrum of the HybridOhmicBath `bath`.
"""
function Gₕ(ω, bath::HybridOhmicBath)
    η = bath.η
    S0 = η / bath.β
    if isapprox(ω, 0, atol=1e-8)
        return 4 / S0
    else
        γ² = (S0 / 2)^2
        return η * ω * exp(-abs(ω) / bath.ωc) / (1 - exp(-bath.β * ω)) / (ω^2 + γ²)
    end
end

"""
    Gₗ(ω, bath::HybridOhmicBath)

Low frequency noise specturm of the HybridOhmicBath `bath`.
"""
function Gₗ(ω, bath::HybridOhmicBath)
    W² = bath.W^2
    ϵ = bath.ϵl
    sqrt(π / 2 / W²) * exp(-(ω - 4ϵ)^2 / 8 / W²)
end

function spectrum(ω, bath::HybridOhmicBath)
    Gl(x) = Gₗ(x, bath)
    Gh(x) = Gₕ(x, bath)
    integrand(x) = Gl(ω - x) * Gh(x)
    a, b = sort([0.0, bath.ϵl])
    res, err = quadgk(integrand, -Inf, a, b, Inf)
    res / 2 / π
end