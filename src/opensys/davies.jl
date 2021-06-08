import StatsBase: sample, Weights

"""
$(TYPEDEF)

`DaviesGenerator` defines a Davies generator.

# Fields

$(FIELDS)
"""
struct DaviesGenerator <: AbstractLiouvillian
    "System bath coupling operators"
    coupling::AbstractCouplings
    "Spectrum density"
    γ::Any
    "Lambshift spectral density"
    S::Any
end

function (D::DaviesGenerator)(du, ρ, ω_ba, v, s::Real)
    γm = D.γ.(ω_ba)
    sm = D.S.(ω_ba)
    for op in D.coupling(s)
        A = v' * op * v
        davies_update!(du, ρ, A, γm, sm)
    end
end

function (D::DaviesGenerator)(du, ρ, ω_ba, s::Real)
    γm = D.γ.(ω_ba)
    sm = D.S.(ω_ba)
    for op in D.coupling(s)
        davies_update!(du, ρ, op, γm, sm)
    end
end

function davies_update!(du, u, A, γ, S)
    A2 = abs2.(A)
    γA = γ .* A2
    Γ = sum(γA, dims=1)
    dim = size(du, 1)
    for a = 1:dim
        for b = 1:a - 1
            @inbounds du[a, a] += γA[a, b] * u[b, b] - γA[b, a] * u[a, a]
            @inbounds du[a, b] +=
                -0.5 * (Γ[a] + Γ[b]) * u[a, b] +
                γ[1, 1] * A[a, a] * A[b, b] * u[a, b]
        end
        for b = a + 1:dim
            @inbounds du[a, a] += γA[a, b] * u[b, b] - γA[b, a] * u[a, a]
            @inbounds du[a, b] +=
                -0.5 * (Γ[a] + Γ[b]) * u[a, b] +
                γ[1, 1] * A[a, a] * A[b, b] * u[a, b]
        end
    end
    H_ls = Diagonal(sum(S .* A2, dims=1)[1, :])
    axpy!(-1.0im, H_ls * u - u * H_ls, du)
end

function update_cache!(cache, D::DaviesGenerator, ω_ba, v, s::Real)
    len = size(ω_ba, 1)
    γm = D.γ.(ω_ba)
    sm = D.S.(ω_ba)
    for op in D.coupling(s)
        A2 = abs2.(v' * op * v)
        for b = 1:len
            for a = 1:len
                @inbounds cache[b, b] -=
                    A2[a, b] * (0.5 * γm[a, b] + 1.0im * sm[a, b])
            end
        end
    end
end

"""
$(TYPEDEF)

Defines correlated Davies generator

# Fields

$(FIELDS)
"""
struct CorrelatedDaviesGenerator <: AbstractLiouvillian
    "System bath coupling operators"
    coupling::AbstractCouplings
    "Spectrum density"
    γ::Any
    "Lambshift spectral density"
    S::Any
    "Indices to iterate"
    inds::Any
end

function (D::CorrelatedDaviesGenerator)(du, ρ, ω_ba, s::Real)
    for (α, β) in D.inds
        γm = D.γ[α,β].(ω_ba)
        sm = D.S[α,β].(ω_ba)
        Aα = D.coupling[α](s)
        Aβ = D.coupling[β](s)
        correlated_davies_update!(du, ρ, Aα, Aβ, γm, sm)
    end
end

function correlated_davies_update!(du, u, Aα, Aβ, γ, S)
    A2 = transpose(Aα) .* Aβ
    γA = γ .* A2
    Γ = sum(γA, dims=1)
    dim = size(du, 1)
    for a = 1:dim
        for b = 1:a - 1
            @inbounds du[a, a] += γA[a, b] * u[b, b] - γA[b, a] * u[a, a]
            @inbounds du[a, b] +=
                -0.5 * (Γ[a] + Γ[b]) * u[a, b] +
                γ[1, 1] * Aα[a, a] * Aβ[b, b] * u[a, b]
        end
        for b = a + 1:dim
            @inbounds du[a, a] += γA[a, b] * u[b, b] - γA[b, a] * u[a, a]
            @inbounds du[a, b] +=
                -0.5 * (Γ[a] + Γ[b]) * u[a, b] +
                γ[1, 1] * Aα[a, a] * Aβ[b, b] * u[a, b]
        end
    end
    H_ls = Diagonal(sum(S .* A2, dims=1)[1, :])
    axpy!(-1.0im, H_ls * u - u * H_ls, du)
end

"""
$(TYPEDEF)

Defines the one-sided AME Liouvillian operator.

# Fields

$(FIELDS)
"""
struct OneSidedAMELiouvillian <: AbstractLiouvillian
    "System bath coupling operators"
    coupling::AbstractCouplings
    "Spectrum density"
    γ::Any
    "Lambshift spectral density"
    S::Any
    "Indices to iterate"
    inds::Any
end

function (A::OneSidedAMELiouvillian)(dρ, ρ, ω_ba, v, s::Real)
    for (α, β) in A.inds
        γm = A.γ[α,β].(ω_ba)
        sm = A.S[α,β].(ω_ba)
        Aα = v' * A.coupling[α](s) * v
        Λ = (0.5 * γm + 1.0im * sm) .* Aα
        Aβ =  v' * A.coupling[β](s) * v
        𝐊₂ = Aβ * Λ * ρ - Λ * ρ * Aβ
        𝐊₂ = 𝐊₂ + 𝐊₂'
        axpy!(-1.0, 𝐊₂, dρ)
    end
end

function (A::OneSidedAMELiouvillian)(du, u, ω_ba, s::Real)
    for (α, β) in A.inds
        γm = A.γ[α,β].(ω_ba)
        sm = A.S[α,β].(ω_ba)
        Aα = A.coupling[α](s)
        Aβ = A.coupling[β](s)
        Λ = (0.5 * γm + 1.0im * sm) .* Aα
        𝐊₂ = Aβ * Λ * u - Λ * u * Aβ
        𝐊₂ = 𝐊₂ + 𝐊₂'
        axpy!(-1.0, 𝐊₂, du)
    end
end