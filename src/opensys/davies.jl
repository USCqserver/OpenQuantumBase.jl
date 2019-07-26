struct Davies <: AbstractOpenSys
    ops
    γ
    S
end

function (D::Davies)(du, u, p, t)
    w, v = eigen_decomp(p.H)
    ρ = v' * u * v
    H = Diagonal(w)
    dρ = -1.0im * p.tf * (H * ρ - ρ * H)
    ω_ba = repeat(w, 1, length(w))
    ω_ba = transpose(ω_ba) - ω_ba
    γm = p.tf * D.γ.(ω_ba)
    sm = p.tf * D.S.(ω_ba)
    for op in D.ops
        A = v' * op * v
        adiabatic_me_update!(dρ, ρ, A, γm, sm)
    end
    mul!(du, v, dρ * v')
end

function adiabatic_me_update!(du, u, A, γ, S)
    A2 = abs2.(A)
    γA = γ .* A2
    Γ = sum(γA, dims = 1)
    dim = size(du)[1]
    for a in 1:dim
        for b in 1:a - 1
            @inbounds du[a, a] += γA[a, b] * u[b, b] - γA[b, a] * u[a, a]
            @inbounds du[a, b] += -0.5 * (Γ[a] + Γ[b]) * u[a, b] + γ[1, 1] * @inbounds A[a, a] * A[b, b] * u[a, b]
        end
        for b in a + 1:dim
            @inbounds du[a, a] += γA[a, b] * u[b, b] - γA[b, a] * u[a, a]
            @inbounds du[a, b] += -0.5 * (Γ[a] + Γ[b]) * u[a, b] + γ[1, 1] * @inbounds A[a, a] * A[b, b] * u[a, b]
        end
    end
    H_ls = Diagonal(sum(S .* A2, dims = 1)[1,:])
    axpy!(-1.0im, H_ls * u - u * H_ls, du)
end
