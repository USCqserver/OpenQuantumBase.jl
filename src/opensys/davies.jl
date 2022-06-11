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

function (D::DaviesGenerator)(du, ρ, gap_idx::GapIndices, v, s::Real)
    l = size(du, 1)
    # pre-rotate all the system bath coupling operators into the energy eigenbasis
    cs = [v'*c*v for c in D.coupling(s)]
    Hₗₛ = spzeros(ComplexF64, l, l)
    for (w, a, b) in positive_gap_indices(gap_idx)
        g₊ = D.γ(w)
        g₋ = D.γ(-w)
        for c in cs
            L₊ = sparse(a, b, c[a + (b .- 1)*l], l, l)
            L₋ = sparse(b, a, c[b + (a .- 1)*l], l, l)
            LL₊ = L₊'*L₊
            LL₋ = L₋'*L₋
            du .+= g₊*(L₊*ρ*L₊'-0.5*(LL₊*ρ+ρ*LL₊)) + g₋*(L₋*ρ*L₋'-0.5*(LL₋*ρ+ρ*LL₋))
            Hₗₛ += D.S(w)*LL₊ + D.S(-w)*LL₋
        end
    end
    g0 = D.γ(0)
    a, b = zero_gap_indices(gap_idx)
	for c in cs
		L = sparse(a, b, c[a + (b .- 1)*l], l, l)
        LL = L'*L
		du .+= g0*(L*ρ*L'-0.5*(LL*ρ+ρ*LL))
        Hₗₛ += D.S(0)*LL
	end
	du .-= 1.0im * (Hₗₛ*ρ - ρ*Hₗₛ)
end

function (D::DaviesGenerator)(du, ρ, gap_idx::GapIndices, s::Real)
    l = size(du, 1)
    cs = [c for c in D.coupling(s)]
    Hₗₛ = spzeros(ComplexF64, l, l)
    for (w, a, b) in positive_gap_indices(gap_idx)
        g₊ = D.γ(w)
        g₋ = D.γ(-w)
        for c in cs
            L₊ = sparse(a, b, c[a + (b .- 1)*l], l, l)
            L₋ = sparse(b, a, c[b + (a .- 1)*l], l, l)
            LL₊ = L₊'*L₊
            LL₋ = L₋'*L₋
            du .+= g₊*(L₊*ρ*L₊'-0.5*(LL₊*ρ+ρ*LL₊)) + g₋*(L₋*ρ*L₋'-0.5*(LL₋*ρ+ρ*LL₋))
            Hₗₛ += D.S(w)*LL₊ + D.S(-w)*LL₋
        end
    end
    g0 = D.γ(0)
    a, b = zero_gap_indices(gap_idx)
	for c in cs
		L = sparse(a, b, c[a + (b .- 1)*l], l, l)
        LL = L'*L
		du .+= g0*(L*ρ*L'-0.5*(LL*ρ+ρ*LL))
        Hₗₛ += D.S(0)*LL
	end
	du .-= 1.0im * (Hₗₛ*ρ - ρ*Hₗₛ)
end

function update_cache!(cache, D::DaviesGenerator, gap_idx::GapIndices, v, s::Real)
    l = size(cache, 1)
    cs = [v'*c*v for c in D.coupling(s)]
    H_eff = spzeros(ComplexF64, l, l)
    for (w, a, b) in positive_gap_indices(gap_idx)
        g₊ = D.γ(w)
        g₋ = D.γ(-w)
        for c in cs
            L₊ = sparse(a, b, c[a + (b .- 1)*l], l, l)
            L₋ = sparse(b, a, c[b + (a .- 1)*l], l, l)
            LL₊ = L₊'*L₊
            LL₋ = L₋'*L₋
            H_eff -= (1.0im*D.S(w)+0.5*g₊)*LL₊ + (1.0im*D.S(-w)+0.5*g₋)*LL₋
        end
    end
    g0 = D.γ(0)
    a, b = zero_gap_indices(gap_idx)
	for c in cs
		L = sparse(a, b, c[a + (b .- 1)*l], l, l)
        LL = L'*L
        H_eff -= (1.0im*D.S(0)+0.5*g0)*LL
	end
    cache .+= H_eff
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

function (D::CorrelatedDaviesGenerator)(du, ρ, gap_idx::GapIndices, s::Real)
    l = size(du, 1)
    Hₗₛ = spzeros(ComplexF64, l, l)
    for (w, a, b) in positive_gap_indices(gap_idx)
        for (α, β) in D.inds
            g₊ = D.γ[α,β](w)
            g₋ = D.γ[α,β](-w)
            Aα = D.coupling[α](s)
            Aβ = D.coupling[β](s)
            L₊ = sparse(a, b, Aβ[a + (b .- 1)*l], l, l)
            L₊d = sparse(a, b, Aα[a + (b .- 1)*l], l, l)'
            L₋ = sparse(b, a, Aβ[b + (a .- 1)*l], l, l)
            L₋d = sparse(b, a, Aα[b + (a .- 1)*l], l, l)'
            LL₊ = L₊d*L₊
            LL₋ = L₋d*L₋
            du .+= g₊*(L₊*ρ*L₊d-0.5*(LL₊*ρ+ρ*LL₊)) + g₋*(L₋*ρ*L₋d-0.5*(LL₋*ρ+ρ*LL₋))
            Hₗₛ += D.S[α,β](w)*LL₊ + D.S[α,β](-w)*LL₋
        end
    end
    a, b = zero_gap_indices(gap_idx)
	for (α, β) in D.inds
        g0 = D.γ[α,β](0)
        Aα = D.coupling[α](s)
        Aβ = D.coupling[β](s)
		L = sparse(a, b, Aβ[a + (b .- 1)*l], l, l)
        Ld = sparse(a, b, Aα[a + (b .- 1)*l], l, l)'
        LL = Ld*L
		du .+= g0*(L*ρ*Ld-0.5*(LL*ρ+ρ*LL))
        Hₗₛ += D.S[α,β](0)*LL
	end
	du .-= 1.0im * (Hₗₛ*ρ - ρ*Hₗₛ)
end

function (D::CorrelatedDaviesGenerator)(du, ρ, gap_idx::GapIndices, v, s::Real)
    l = size(du, 1)
    Hₗₛ = spzeros(ComplexF64, l, l)
    for (w, a, b) in positive_gap_indices(gap_idx)
        for (α, β) in D.inds
            g₊ = D.γ[α,β](w)
            g₋ = D.γ[α,β](-w)
            Aα = v' * D.coupling[α](s) * v
            Aβ = V' * D.coupling[β](s) * v
            L₊ = sparse(a, b, Aβ[a + (b .- 1)*l], l, l)
            L₊d = sparse(a, b, Aα[a + (b .- 1)*l], l, l)'
            L₋ = sparse(b, a, Aβ[b + (a .- 1)*l], l, l)
            L₋d = sparse(b, a, Aα[b + (a .- 1)*l], l, l)'
            LL₊ = L₊d*L₊
            LL₋ = L₋d*L₋
            du .+= g₊*(L₊*ρ*L₊d-0.5*(LL₊*ρ+ρ*LL₊)) + g₋*(L₋*ρ*L₋d-0.5*(LL₋*ρ+ρ*LL₋))
            Hₗₛ += D.S[α,β](w)*LL₊ + D.S[α,β](-w)*LL₋
        end
    end
    a, b = zero_gap_indices(gap_idx)
	for (α, β) in D.inds
        g0 = D.γ[α,β](0)
        Aα = v' * D.coupling[α](s) * v
        Aβ = v' * D.coupling[β](s) * v
		L = sparse(a, b, Aβ[a + (b .- 1)*l], l, l)
        Ld = sparse(a, b, Aα[a + (b .- 1)*l], l, l)'
        LL = Ld*L
		du .+= g0*(L*ρ*Ld-0.5*(LL*ρ+ρ*LL))
        Hₗₛ += D.S[α,β](0)*LL
	end
	du .-= 1.0im * (Hₗₛ*ρ - ρ*Hₗₛ)
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

function (A::OneSidedAMELiouvillian)(dρ, ρ, g_idx::GapIndices, v, s::Real)
    ω_ba = gap_matrix(g_idx)
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

function (A::OneSidedAMELiouvillian)(du, u, g_idx::GapIndices, s::Real)
    ω_ba = gap_matrix(g_idx)
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