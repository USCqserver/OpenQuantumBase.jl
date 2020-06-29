import StatsBase: sample, Weights

"""
$(TYPEDEF)

Defines Davies generator

# Fields

$(FIELDS)
"""
struct DaviesGenerator <: AbstractOpenSys
    """System bath coupling operators"""
    coupling::AbstractCouplings
    """Spectrum density"""
    γ::Any
    """Lambshift spectrum density"""
    S::Any
end


function (D::DaviesGenerator)(du, ρ, ω_ba, v, tf::Real, t::Real)
    γm = tf * D.γ.(ω_ba)
    sm = tf * D.S.(ω_ba)
    for op in D.coupling(t)
        A = v' * (op * v)
        davies_update!(du, ρ, A, γm, sm)
    end
end

(D::DaviesGenerator)(du, ρ, ω_ba, v, tf::UnitTime, t::Real) =
    D(du, ρ, ω_ba, v, 1.0, t / tf)


function (D::DaviesGenerator)(du, u, ω_ba, tf::Real, t::Real)
    γm = tf * D.γ.(ω_ba)
    sm = tf * D.S.(ω_ba)
    for op in D.coupling(t)
        davies_update!(du, u, op, γm, sm)
    end
end


(D::DaviesGenerator)(du, u, ω_ba, tf::UnitTime, t::Real) =
    D(du, u, ω_ba, 1.0, t / tf)


function davies_update!(du, u, A, γ, S)
    A2 = abs2.(A)
    γA = γ .* A2
    Γ = sum(γA, dims = 1)
    dim = size(du, 1)
    for a = 1:dim
        for b = 1:a-1
            du[a, a] += γA[a, b] * u[b, b] - γA[b, a] * u[a, a]
            du[a, b] +=
                -0.5 * (Γ[a] + Γ[b]) * u[a, b] +
                γ[1, 1] * A[a, a] * A[b, b] * u[a, b]
        end
        for b = a+1:dim
            du[a, a] += γA[a, b] * u[b, b] - γA[b, a] * u[a, a]
            du[a, b] +=
                -0.5 * (Γ[a] + Γ[b]) * u[a, b] +
                γ[1, 1] * A[a, a] * A[b, b] * u[a, b]
        end
    end
    H_ls = Diagonal(sum(S .* A2, dims = 1)[1, :])
    axpy!(-1.0im, H_ls * u - u * H_ls, du)
end


"""
$(TYPEDEF)

Defines an adiabatic master equation differential equation operator.

# Fields

$(FIELDS)
"""
struct AMEDiffEqOperator{is_sparse,is_adiabatic_frame}
    """Hamiltonian"""
    H::AbstractHamiltonian
    """Davies generator"""
    Davies::DaviesGenerator
    """Number of levels to keep"""
    lvl::Int
    """Internal cache"""
    u_cache::Matrix{ComplexF64}
    """Eigen decomposition function"""
    _eig::Any
end

eigen_decomp(A::AMEDiffEqOperator, t) = A._eig(A.H, t, A.lvl)

function AMEDiffEqOperator(
    H::AbstractHamiltonian,
    davies,
    lvl::Int,
    eig_init = EIGEN_DEFAULT;
    kwargs...,
)
    # initialze the internal cache
    # the internal cache will also be dense for current version
    u_cache = Matrix{eltype(H)}(undef, lvl, lvl)
    # check if the Hamiltonian is sparse
    is_sparse = typeof(H) <: AbstractSparseHamiltonian
    # check if the Hamiltonian is defined in adiabatic frames
    is_adiabatic_frame = typeof(H) <: AdiabaticFrameHamiltonian
    # initialze the eigen decomposition method
    eig = eig_init(H)
    # return AMEDiffEqOperator
    AMEDiffEqOperator{is_sparse,is_adiabatic_frame}(
        H,
        davies,
        lvl,
        u_cache,
        eig,
    )
end

(A::AMEDiffEqOperator{S,false})(
    du,
    u,
    p::ODEParams{T},
    t,
) where {S,T<:AbstractFloat} = ame_update!(du, u, p.tf, t, A)

(A::AMEDiffEqOperator{S,false})(
    du,
    u,
    p::ODEParams{T},
    t,
) where {S,T<:UnitTime} = ame_update!(du, u, 1.0, t / p.tf, A)


function ame_update!(du, u, tf, t, A)
    w, v = eigen_decomp(A, t)
    ρ = v' * u * v
    H = Diagonal(w)
    A.u_cache .= -1.0im * tf * (H * ρ - ρ * H)
    ω_ba = transpose(w) .- w
    A.Davies(A.u_cache, ρ, ω_ba, v, tf, t)
    mul!(du, v, A.u_cache * v')
end


function (A::AMEDiffEqOperator{S,true})(du, u, p, t) where {S}
    A.H(du, u, p.tf, t)
    ω_ba = ω_matrix(A.H, A.lvl)
    A.Davies(du, u, ω_ba, p.tf, t)
end


# struct AFRWADiffEqOperator{control_type}
#     H::Any
#     Davies::Any
#     lvl::Int
# end
#
#
# function AFRWADiffEqOperator(H, davies; lvl = nothing, control = nothing)
#     if lvl == nothing
#         lvl = H.size[1]
#     end
#     AFRWADiffEqOperator{typeof(control)}(H, davies, lvl)
# end
#
#
# function (D::AFRWADiffEqOperator{Nothing})(du, u, p, t)
#     w, v = ω_matrix_RWA(D.H, p.tf, t, D.lvl)
#     ρ = v' * u * v
#     H = Diagonal(w)
#     cache = diag_update(H, ρ, p.tf)
#     ω_ba = repeat(w, 1, length(w))
#     ω_ba = transpose(ω_ba) - ω_ba
#     D.Davies(cache, ρ, ω_ba, v, p.tf, t)
#     mul!(du, v, cache * v')
# end
#
# @inline diag_update(H, ρ, tf::Real) = -1.0im * tf * (H * ρ - ρ * H)
#
# @inline diag_update(H, ρ, tf::UnitTime) = -1.0im * (H * ρ - ρ * H)


"""
$(TYPEDEF)

Defines an adiabatic master equation trajectory operator. The object is used to update the cache for `DiffEqArrayOperator`.

# Fields

$(FIELDS)
"""
struct AMETrajectoryOperator{is_sparse,is_adiabatic_frame} <:
       AbstractAnnealingControl
    """Hamiltonian"""
    H::Any
    """Davies generator"""
    Davies::DaviesGenerator
    """Number of levels to keep"""
    lvl::Int
    """Internal cache"""
    u_cache::Vector{ComplexF64}
    """Eigen decomposition function"""
    _eig::Any
end

eigen_decomp(A::AMETrajectoryOperator, t) = A._eig(A.H, t, A.lvl)

function AMETrajectoryOperator(
    H,
    davies,
    lvl::Int,
    eig_init = EIGEN_DEFAULT;
    kwargs...,
)
    # initialze the internal cache
    # the internal cache will also be dense for current version
    u_cache = Vector{eltype(H)}(undef, lvl)
    # check if the Hamiltonian is sparse
    is_sparse = typeof(H) <: AbstractSparseHamiltonian
    # check if the Hamiltonian is defined in adiabatic frames
    is_adiabatic_frame = typeof(H) <: AdiabaticFrameHamiltonian
    # initialze the eigen decomposition method
    eig = eig_init(H)
    AMETrajectoryOperator{is_sparse,is_adiabatic_frame}(
        H,
        davies,
        lvl,
        u_cache,
        eig,
    )
end


function update_cache!(cache, A::AMETrajectoryOperator, tf::Real, s::Real)
    w, v = eigen_decomp(A, s)
    ω_ba = transpose(w) .- w
    internal_cache = mul!(A.u_cache, -1.0im * tf, w)
    γm = tf * A.Davies.γ.(ω_ba)
    sm = tf * A.Davies.S.(ω_ba)
    for op in A.Davies.coupling(s)
        A2 = abs2.(v' * op * v)
        for b = 1:A.lvl
            for a = 1:A.lvl
                @inbounds internal_cache[b] -=
                    A2[a, b] * (0.5 * γm[a, b] + 1.0im * sm[a, b])
            end
        end
    end
    cache .= v * Diagonal(internal_cache) * v'
end


update_cache!(cache, A::AMETrajectoryOperator, tf::UnitTime, t::Real) =
    update_cache!(cache, A, 1.0, t / tf)

"""
    ame_jump(A::AMETrajectoryOperator, u, tf, s::Real)

Calculate the jump operator for the AMETrajectoryOperator at time `s`.
"""
function ame_jump(A::AMETrajectoryOperator, u, tf::Real, s::Real)
    w, v = eigen_decomp(A, s)
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
        res =
            sqrt(γm[choice[2], choice[3]]) *
            σab[choice[1]][choice[2], choice[3]] *
            v[:, choice[2]] *
            v[:, choice[3]]'
    end
    res
end


@inline ame_jump(A::AMETrajectoryOperator, u, tf::UnitTime, t::Real) =
    ame_jump(A, u, 1.0, t / tf)
