"""
$(TYPEDEF)

An custum bath object defined by the two-point correlation function and the corresponding spectrum.

$(FIELDS)
"""
mutable struct CustomBath <: AbstractBath
    "correlation function"
    cfun::Any
    "spectrum"
    γ::Any
end

CustomBath(; correlation=nothing, spectrum=nothing) =
    CustomBath(correlation, spectrum)
correlation(τ, bath::CustomBath) = bath.cfun(τ)
spectrum(ω, bath::CustomBath) = bath.γ(ω)
γ(ω, bath::CustomBath) = bath.γ(ω)
S(w, bath::CustomBath; atol=1e-7) =
    lambshift(w, (ω) -> spectrum(ω, bath), atol=atol)

function build_correlation(bath::CustomBath)
    bath.cfun == nothing && error("Correlation function is not specified.")
    if numargs(bath.cfun) == 1
        SingleFunctionMatrix((t₁, t₂) -> bath.cfun(t₁ - t₂))
    else
        SingleFunctionMatrix(bath.cfun)
    end
end

build_spectrum(bath::CustomBath) =
    bath.γ == nothing ? error("Noise spectrum is not specified.") : bath.γ

"""
$(TYPEDEF)

`CorrelatedBath` defines a correlated bath type by the matrices of its two-point correlation functions and corresponding spectrums.

$(FIELDS)
"""
mutable struct CorrelatedBath <: AbstractBath
    "correlation function"
    cfun::Any
    "spectrum"
    γ::Any
    "bath correlator idx"
    inds::Any
end

CorrelatedBath(inds; correlation=nothing, spectrum=nothing) = CorrelatedBath(correlation, spectrum, inds)
build_correlation(bath::CorrelatedBath) = bath.cfun == nothing ? throw(ArgumentError("Correlation function is not specified.")) : bath.cfun
build_spectrum(bath::CorrelatedBath) = bath.γ == nothing ? error("Spectrum is not specified.") : bath.γ
build_inds(bath::CorrelatedBath) = bath.inds

function build_lambshift(ω_range::AbstractVector, turn_on::Bool, bath::CorrelatedBath, lambshift_S)
    gamma = build_spectrum(bath)
    l = size(gamma)
    if turn_on == true
        if lambshift_S == nothing
            if isempty(ω_range)
                S_loc = Array{Function,2}(undef, l)
                for i in eachindex(gamma)
                    S_loc[i] = (w) -> lamb(w, gamma[i])
                end
            else
                S_loc = Array{Any,2}(undef, l)
                for i in eachindex(gamma)
                    s_list = [lambshift(ω, gamma[i]) for ω in ω_range]
                    S_loc[i] = construct_interpolations(ω_range, s_list)
                end
            end
        else
            S_loc = lambshift_S
        end
    else
        S_loc = Array{Function,2}(undef, size(gamma))
        S_loc .= (ω) -> 0.0
    end
    S_loc
end

"""
$(TYPEDEF)

A bath object to hold jump correlator of ULE.

$(FIELDS)
"""
struct ULEBath <: AbstractBath
    "correlation function"
    cfun::Any
end

function build_jump_correlation(bath::ULEBath)
    if numargs(bath.cfun) == 1
        SingleFunctionMatrix((t₁, t₂) -> bath.cfun(t₁ - t₂))
    else
        SingleFunctionMatrix(bath.cfun)
    end 
end