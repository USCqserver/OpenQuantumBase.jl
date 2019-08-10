"""
$(TYPEDEF)

Defines Davies generator

# Fields

$(FIELDS)
"""
struct DaviesGenerator <: AbstractOpenSys
    coupling
    γ
    S
end


function (D::DaviesGenerator)(du, u, ω_ba, v, tf, t)
    ρ = v' * u * v
    γm = tf * D.γ.(ω_ba)
    sm = tf * D.S.(ω_ba)
    for op in D.coupling(t)
        A = v' * (op * v)
        adiabatic_me_update!(du, ρ, A, γm, sm)
    end
end


function (D::DaviesGenerator)(du, u, ω_ba, tf, t)
    γm = tf * D.γ.(ω_ba)
    sm = tf * D.S.(ω_ba)
    for op in D.coupling(t)
        adiabatic_me_update!(du, u, op, γm, sm)
    end
end


struct AMEDiffEqOperator{AF}
    H
    Davies
    lvl
end


function AMEDiffEqOperator(H, coupling, γ, S; lvl=nothing)
    if typeof(H) <: AdiabaticFrameHamiltonian
        AMEDiffEqOperator{true}(H, coupling, γ, S)
    else
        AMEDiffEqOperator{false}(H, coupling, γ, S)
    end
end

function (D::AMEDiffEqOperator{true})(du, u, p, t)
    D.H(du, u, p.tf, t)
    ω_ba = ω_matrix(D.H)
    γm = p.tf * D.γ.(ω_ba)
    sm = p.tf * D.S.(ω_ba)
    for op in D.coupling(t)
        adiabatic_me_update!(du, u, op, γm, sm)
    end
end


function (D::AMEDiffEqOperator{false})(du, u, p, t)
    cache = D.H(t)
    w, v = ode_eigen_decomp(D.H, D.L)
    ρ = v' * u * v
    H = Diagonal(w)
    fill!(cache, 0.0im)
    comm_update!(cache, H, u, p.tf)
    ω_ba = repeat(w, 1, length(w))
    ω_ba = transpose(ω_ba) - ω_ba
    γm = p.tf * D.γ.(ω_ba)
    sm = p.tf * D.S.(ω_ba)
    for op in D.coupling(t)
        # the parenthesis here is for sparse op
        A = v' * (op * v)
        adiabatic_me_update!(cache, ρ, A, γm, sm)
    end
    mul!(du, v, cache * v')
end


function adiabatic_me_update!(du, u, A, γ, S)
    A2 = abs2.(A)
    γA = γ .* A2
    Γ = sum(γA, dims = 1)
    dim = size(du)[1]
    for a in 1:dim
        for b in 1:a - 1
            du[a, a] += γA[a, b] * u[b, b] - γA[b, a] * u[a, a]
            du[a, b] += -0.5 * (Γ[a] + Γ[b]) * u[a, b] + γ[1, 1] * A[a, a] * A[b, b] * u[a, b]
        end
        for b in a + 1:dim
            du[a, a] += γA[a, b] * u[b, b] - γA[b, a] * u[a, a]
            du[a, b] += -0.5 * (Γ[a] + Γ[b]) * u[a, b] + γ[1, 1] * A[a, a] * A[b, b] * u[a, b]
        end
    end
    H_ls = Diagonal(sum(S .* A2, dims = 1)[1,:])
    axpy!(-1.0im, H_ls * u - u * H_ls, du)
end
