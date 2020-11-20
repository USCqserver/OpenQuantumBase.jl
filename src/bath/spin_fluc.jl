import Distributions: Exponential, product_distribution

"""
$(TYPEDEF)

A symetric random telegraph noise with switch rate `γ/2` magnitude `b`

$(FIELDS)
"""
struct SymetricRTN
    "Magnitude"
    b::Any
    "Two times the switching probability"
    γ::Any
end

correlation(τ, R::SymetricRTN) = R.b^2 * exp(-R.γ * τ)
spectrum(ω, R::SymetricRTN) = 2 * R.b^2 * R.γ / (ω^2 + R.γ^2)
construct_distribution(R::SymetricRTN) = Exponential(1 / R.γ)

"""
$(TYPEDEF)

An ensemble of random telegraph noise.

$(FIELDS)
"""
struct EnsembleFluctuator{T} <: StochasticBath
    """A list of RTNs"""
    f::Vector{T}
end

EnsembleFluctuator(b::AbstractArray{T}, ω::AbstractArray{T}) where {T <: Number} =
    EnsembleFluctuator([SymetricRTN(x, y) for (x, y) in zip(b, ω)])
correlation(τ, E::EnsembleFluctuator) = sum((x) -> correlation(τ, x), E.f)
spectrum(ω, E::EnsembleFluctuator) = sum((x) -> spectrum(ω, x), E.f)
construct_distribution(E::EnsembleFluctuator) =
    product_distribution([construct_distribution(x) for x in E.f])

Base.length(E::EnsembleFluctuator) = Base.length(E.f)
Base.show(io::IO, ::MIME"text/plain", E::EnsembleFluctuator) =
    print(io, "Fluctuator ensemble with ", length(E), " fluctuators")
Base.show(io::IO, E::EnsembleFluctuator) =
    print(io, "Fluctuator ensemble with ", length(E), " fluctuators")
