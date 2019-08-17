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


function (D::DaviesGenerator)(du, ρ, ω_ba, v, tf::Real, t::Real)
    γm = tf * D.γ.(ω_ba)
    sm = tf * D.S.(ω_ba)
    for op in D.coupling(t)
        A = v' * (op * v)
        adiabatic_me_update!(du, ρ, A, γm, sm)
    end
end


function (D::DaviesGenerator)(du, ρ, ω_ba, v, tf::UnitTime, t::Real)
    γm = D.γ.(ω_ba)
    sm = D.S.(ω_ba)
    for op in D.coupling(t / tf)
        A = v' * (op * v)
        adiabatic_me_update!(du, ρ, A, γm, sm)
    end
end


function (D::DaviesGenerator)(du, u, ω_ba, tf::Real, t::Real)
    γm = tf * D.γ.(ω_ba)
    sm = tf * D.S.(ω_ba)
    for op in D.coupling(t)
        adiabatic_me_update!(du, u, op, γm, sm)
    end
end


function (D::DaviesGenerator)(du, u, ω_ba, tf::UnitTime, t::Real)
    γm = D.γ.(ω_ba)
    sm = D.S.(ω_ba)
    for op in D.coupling(t / tf)
        adiabatic_me_update!(du, u, op, γm, sm)
    end
end


struct AMEDiffEqOperator{AF,control_type}
    H
    Davies
    lvl::Int
end


function AMEDiffEqOperator(H, davies; lvl = nothing, control = nothing)
    if lvl == nothing
        lvl = H.size[1]
    end
    if typeof(H) <: AbstractSparseHamiltonian
        if H.size[1] == 2
            @warn "Hamiltonian size is too small for sparse factorization. Convert to dense Hamiltonian"
            H = to_dense(A.H)
        elseif lvl == H.size[1]
            @warn "Sparse Hamiltonian detected. Truncate the level to n-1."
            lvl = lvl - 1
        end
    end
    if typeof(H) <: AdiabaticFrameHamiltonian
        AF = true
    else
        AF = false
    end
    AMEDiffEqOperator{AF,typeof(control)}(H, davies, lvl)
end


function (D::AMEDiffEqOperator{true,Nothing})(du, u, p, t)
    D.H(du, u, p.tf, t)
    ω_ba = ω_matrix(D.H, D.lvl)
    D.Davies(du, u, ω_ba, p.tf, t)
end


function (D::AMEDiffEqOperator{false,Nothing})(du, u, p, t)
    cache = D.H(t)
    w, v = ode_eigen_decomp(D.H, D.lvl)
    ρ = v' * u * v
    H = Diagonal(w)
    diag_cache_update!(cache, H, ρ, p.tf)
    ω_ba = repeat(w, 1, length(w))
    ω_ba = transpose(ω_ba) - ω_ba
    D.Davies(cache, ρ, ω_ba, v, p.tf, t)
    mul!(du, v, cache * v')
end


function adiabatic_me_update!(du, u, A, γ, S)
    A2 = abs2.(A)
    γA = γ .* A2
    Γ = sum(γA, dims = 1)
    dim = size(du)[1]
    for a in 1:dim
        for b in 1:a-1
            du[a, a] += γA[a, b] * u[b, b] - γA[b, a] * u[a, a]
            du[a, b] += -0.5 * (Γ[a] + Γ[b]) * u[a, b] + γ[1, 1] * A[a, a] * A[b, b] * u[a, b]
        end
        for b in a+1:dim
            du[a, a] += γA[a, b] * u[b, b] - γA[b, a] * u[a, a]
            du[a, b] += -0.5 * (Γ[a] + Γ[b]) * u[a, b] + γ[1, 1] * A[a, a] * A[b, b] * u[a, b]
        end
    end
    H_ls = Diagonal(sum(S .* A2, dims = 1)[1, :])
    axpy!(-1.0im, H_ls * u - u * H_ls, du)
end


@inline function diag_cache_update!(cache, H, ρ, tf::Real)
    cache .= -1.0im * tf * (H * ρ - ρ * H)
end

@inline function diag_cache_update!(cache, H, ρ, tf::UnitTime)
    cache .= -1.0im * (H * ρ - ρ * H)
end


struct AFRWADiffEqOperator{control_type}
    H
    Davies
    lvl::Int
end


function AFRWADiffEqOperator(H, davies; lvl = nothing, control = nothing)
    if lvl == nothing
        lvl = H.size[1]
    end
    AFRWADiffEqOperator{typeof(control)}(H, davies, lvl)
end


function (D::AFRWADiffEqOperator{Nothing})(du, u, p, t)
    w, v = ω_matrix_RWA(D.H, p.tf, t, D.lvl)
    ρ = v' * u * v
    H = Diagonal(w)
    cache = diag_update(H, ρ, p.tf)
    ω_ba = repeat(w, 1, length(w))
    ω_ba = transpose(ω_ba) - ω_ba
    D.Davies(cache, ρ, ω_ba, v, p.tf, t)
    mul!(du, v, cache * v')
end

@inline diag_update(H, ρ, tf::Real) = -1.0im * tf * (H * ρ - ρ * H)

@inline diag_update(H, ρ, tf::UnitTime) = -1.0im * (H * ρ - ρ * H)
