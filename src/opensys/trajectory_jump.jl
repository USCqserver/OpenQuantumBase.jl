function lind_jump(lind::LindbladLiouvillian, u, p, s::Real)
    l = length(lind)
    prob = Float64[]
    ops = Vector{Matrix{ComplexF64}}()
    for (γfun, Lfun) in zip(lind.γ, lind.L)
        L = Lfun(s)
        γ = γfun(s)
        push!(prob, γ * norm(L * u))
        push!(ops, L)
    end
    sample(ops, Weights(prob))
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
            for a = 1:b - 1
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
        for i in range(1, stop=sys_dim)
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

# TODO: Better implemention of ame_jump function
"""
    lindblad_jump(A::DiffEqLiouvillian, u, p, t::Real)

Calculate the jump operator for the `DiffEqLiouvillian` at time `t`.
"""
function lindblad_jump(Op::DiffEqLiouvillian{true,false}, u, p, t::Real)
    s = p(t)
    w, v = Op.H.EIGS(Op.H, s, Op.lvl)
    ω_ba = transpose(w) .- w
    resample([ame_jump(x, u, ω_ba, v, s) for x in Op.opensys_eig], u)
end

function lindblad_jump(Op::DiffEqLiouvillian{false,false}, u, p, t::Real)
    s = p(t)
    jump_ops = resample([lind_jump(x, u, p, s) for x in Op.opensys], u)
end

function resample(Ls, u)
    if length(Ls) == 1
        Ls[1]
    else
        prob = [norm(L * u) for L in Ls]
        sample(Ls, Weights(prob))
    end
end