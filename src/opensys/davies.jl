"""
$(TYPEDEF)

Defines Davies generator

# Fields

$(FIELDS)
"""
struct DaviesGenerator <: AbstractOpenSys
    """System bath coupling operators"""
    coupling
    """Spectrum density"""
    γ
    """Lambshift spectrum density"""
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


@inline ame_jump(w, v, u, D::DaviesGenerator, tf::UnitTime, t::Real) = ame_jump(w, v, u, D, tf, t/tf)


"""
$(TYPEDEF)

Defines an adiabatic master equation differential equation operator.

# Fields

$(FIELDS)
"""
struct AMEDiffEqOperator{adiabatic_frame,control_type}
    """Hamiltonian"""
    H
    """Davies generator"""
    Davies::DaviesGenerator
    """Number of levels to keep"""
    lvl::Int
    """Internal cache"""
    u_cache
end


function AMEDiffEqOperator(H, davies; lvl = nothing, control = nothing)
    if lvl == nothing
        lvl = H.size[1]
    end
    # for sparse matrix:
    # 1. the eigen decomposition does not work with 2X2 matrices
    # 2. the level must be at most total level-1
    if typeof(H) <: AbstractSparseHamiltonian
        if H.size[1] == 2
            @warn "Hamiltonian size is too small for sparse factorization. Convert to dense Hamiltonian"
            H = to_dense(A.H)
        elseif lvl == H.size[1]
            @warn "Sparse Hamiltonian detected. Truncate the level to n-1."
            lvl = lvl - 1
        end
    end
    # initialze the internal cache
    # the internal cache will also be dense at current stage
    u_cache = Matrix{eltype(H)}(undef, lvl, lvl)
    # check if the Hamiltonian is defined in adiabatic frames
    if typeof(H) <: AdiabaticFrameHamiltonian
        adiabatic_frame = true
    else
        adiabatic_frame = false
    end
    AMEDiffEqOperator{adiabatic_frame,typeof(control)}(H, davies, lvl, u_cache)
end


function (D::AMEDiffEqOperator{true,Nothing})(du, u, p, t)
    D.H(du, u, p.tf, t)
    ω_ba = ω_matrix(D.H, D.lvl)
    D.Davies(du, u, ω_ba, p.tf, t)
end


function (D::AMEDiffEqOperator{false,Nothing})(du, u, p, t)
    hmat = D.H(t)
    w, v = ode_eigen_decomp(D.H, D.lvl)
    ρ = v' * u * v
    H = Diagonal(w)
    diag_cache_update!(D.u_cache, H, ρ, p.tf)
    ω_ba = repeat(w, 1, length(w))
    ω_ba = transpose(ω_ba) - ω_ba
    D.Davies(D.u_cache, ρ, ω_ba, v, p.tf, t)
    mul!(du, v, D.u_cache * v')
end


@inline function diag_cache_update!(cache, H, ρ, tf::Real)
    cache .= -1.0im * tf * (H * ρ - ρ * H)
end


@inline function diag_cache_update!(cache, H, ρ, tf::UnitTime)
    cache .= -1.0im * (H * ρ - ρ * H)
end


function adiabatic_me_update!(du, u, A, γ, S)
    A2 = abs2.(A)
    γA = γ .* A2
    Γ = sum(γA, dims = 1)
    dim = size(du, 1)
    for a = 1:dim
        for b = 1:a-1
            du[a, a] += γA[a, b] * u[b, b] - γA[b, a] * u[a, a]
            du[a, b] += -0.5 * (Γ[a] + Γ[b]) * u[a, b] +
                        γ[1, 1] * A[a, a] * A[b, b] * u[a, b]
        end
        for b = a+1:dim
            du[a, a] += γA[a, b] * u[b, b] - γA[b, a] * u[a, a]
            du[a, b] += -0.5 * (Γ[a] + Γ[b]) * u[a, b] +
                        γ[1, 1] * A[a, a] * A[b, b] * u[a, b]
        end
    end
    H_ls = Diagonal(sum(S .* A2, dims = 1)[1, :])
    axpy!(-1.0im, H_ls * u - u * H_ls, du)
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


"""
$(TYPEDEF)

Defines an adiabatic master equation trajectory operator. The object is used to update the cache for
DiffEqArrayOperator.

# Fields

$(FIELDS)
"""
struct AMETrajectoryOperator <: AbstractAnnealingControl
    """Hamiltonian"""
    H
    """Davies generator"""
    Davies::DaviesGenerator
    """Number of levels to keep"""
    lvl::Int
end


function update_cache!(cache, A::AMETrajectoryOperator, tf::Real, s::Real)
    A.H(s)
    w, v = ode_eigen_decomp(A.H, A.lvl)
    ω_ba = transpose(w) .- w
    internal_cache = -1.0im * tf * w
    γm = tf * A.Davies.γ.(ω_ba)
    sm = tf * A.Davies.S.(ω_ba)
    for op in A.Davies.coupling(s)
        A2 = abs2.(v' * op * v)
        for b = 1:A.lvl
            for a = 1:A.lvl
                @inbounds internal_cache[b] -= A2[a, b] * (0.5 * γm[a, b] + 1.0im * sm[a, b])
            end
        end
    end
    cache .= v * Diagonal(internal_cache) * v'
end


"""
    ame_jump(A::AMETrajectoryOperator, u, tf, s::Real)

Calculate the jump operator for the AMETrajectoryOperator at time `s`.
"""
function ame_jump(A::AMETrajectoryOperator, u, tf::Real, s::Real)
    A.H(s)
    w, v = ode_eigen_decomp(A.H, A.lvl)
    # calculate all dimensions
    sys_dim = size(A.H, 1)
    num_noise = length(A.Davies.coupling)
    prob_dim = (A.lvl * (A.lvl - 1) + 1) * num_noise
    ω_ba = transpose(w) .- w
    γm = A.Davies.γ.(ω_ba)
    prob = Array{Float64,1}(undef, prob_dim)
    tag = Array{Tuple{Int,Int,Int},1}(undef, prob_dim)
    idx = 1
    ϕb = abs2.(v' * u)
    σab = [v' * op * v for op in A.Davies.coupling(s)]
    for i = 1:num_noise
        γA = γm .* abs2.(σab[i])
        for b = 1:A.lvl
            for a = 1:b-1
                prob[idx] = γA[a, b] * ϕb[b]
                tag[idx] = (i, a, b)
                idx += 1
                prob[idx] = γA[b, a] * ϕb[a]
                tag[idx] = (i, b, a)
                idx += 1
            end
        end
        prob[idx] = transpose(diag(γA)) * ϕb
        tag[idx] = (i, 0, 0)
        idx += 1
    end
    choice = sample(tag, Weights(prob))
    if choice[2] == 0
        res = zeros(ComplexF64, sys_dim, sys_dim)
        for i in range(1, stop = sys_dim)
            res += sqrt(γm[1, 1]) * σab[choice[1]][i, i] * v[:, i] * v[:, i]'
        end
    else
        res = sqrt(γm[choice[2], choice[3]]) *
              σab[choice[1]][choice[2], choice[3]] * v[:, choice[2]] *
              v[:, choice[3]]'
    end
    res
end


@inline ame_jump(A::AMETrajectoryOperator, u, tf::UnitTime, t::Real) = ame_jump(A, u, tf, t/tf)
