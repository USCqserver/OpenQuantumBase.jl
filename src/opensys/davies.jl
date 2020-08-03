import StatsBase: sample, Weights
AMEOperator(H::AbstractHamiltonian, D, lvl::Int) = OpenSysOp(H, D, lvl, true)

"""
$(TYPEDEF)

Defines Davies generator

# Fields

$(FIELDS)
"""
struct DaviesGenerator <: AbstractLiouvillian
    """System bath coupling operators"""
    coupling::AbstractCouplings
    """Spectrum density"""
    γ::Any
    """Lambshift spectrum density"""
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

function ame_jump(D::DaviesGenerator, u, ω_ba, v, s)
    lvl = size(ω_ba, 1)
    sys_dim = size(v, 1)
    num_noise = length(D.coupling)
    prob_dim = (lvl * (lvl - 1) + 1) * num_noise
    γm = D.γ.(ω_ba)
    prob = Array{Float64,1}(undef, prob_dim)
    tag = Array{Tuple{Int,Int,Int},1}(undef, prob_dim)
    idx = 1
    ϕb = abs2.(v' * u)
    σab = [v' * op * v for op in D.coupling(s)]
    for i = 1:num_noise
        γA = γm .* abs2.(σab[i])
        for b = 1:lvl
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

#TODO: Better implemention of ame_jump function
"""
    ame_jump(A::OpenSysOp, u, p, t::Real)

Calculate the jump operator for the `OpenSysOp` at time `t`.
"""
function ame_jump(Op::OpenSysOp{true,false}, u, p, t::Real)
    s = p(t)
    w, v = Op.H.EIGS(Op.H, s, Op.lvl)
    ω_ba = transpose(w) .- w
    sum((x) -> ame_jump(x, u, ω_ba, v, s), Op.opensys)
end

function ame_jump(Op::OpenSysOpHybrid{false}, u, p, t::Real)
    s = p(t)
    hmat = Op.H(s)
    w, v = Op.H.EIGS(Op.H, s, Op.lvl)
    ω_ba = transpose(w) .- w
    sum((x) -> ame_jump(x, u, ω_ba, v, s), Op.opensys_eig)
end
