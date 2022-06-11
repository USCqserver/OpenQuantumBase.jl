
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

function ame_jump(D::DaviesGenerator, u, gap_idx::GapIndices, v, s)
    l = get_lvl(gap_idx)
    prob_dim = get_gaps_num(gap_idx) * length(D.coupling)
    prob = Array{Float64,1}(undef, prob_dim)
    tag = Array{Tuple{Int,Vector{Int},Vector{Int},Float64},1}(undef, prob_dim)
    idx = 1
    ϕb = v' * u
    σab = [v' * op * v for op in D.coupling(s)]
    for (w, a, b) in positive_gap_indices(gap_idx)
        g₊ = D.γ(w)
        g₋ = D.γ(-w)
        for i in eachindex(σab)
            L₊ = sparse(a, b, σab[i][a + (b .- 1)*l], l, l)
            L₋ = sparse(b, a, σab[i][b + (a .- 1)*l], l, l)
            ϕ₊ = L₊ * ϕb
            prob[idx] = g₊ * real(ϕ₊' * ϕ₊)
            tag[idx] = (i, a, b, sqrt(g₊))
            idx += 1
            ϕ₋ = L₋ * ϕb
            prob[idx] = g₋ * real(ϕ₋' * ϕ₋)
            tag[idx] = (i, b, a, sqrt(g₋))
            idx += 1
        end 
    end
    g0 = D.γ(0)
    a, b = zero_gap_indices(gap_idx)
	for i in eachindex(σab)
		L = sparse(a, b, σab[i][a + (b .- 1)*l], l, l)
        ϕ = L * ϕb
        prob[idx] = g0 * (ϕ' * ϕ)
        tag[idx] = (i, a, b, sqrt(g0))
        idx += 1
	end
    choice = sample(tag, Weights(prob))
    L = choice[4] * sparse(choice[2], choice[3], σab[choice[1]][choice[2] + (choice[3] .- 1)*l] , l, l)

    v * L * v'
end

# TODO: Better implemention of ame_jump function
"""
$(SIGNATURES)

Calculate the jump operator for the `DiffEqLiouvillian` at time `t`.
"""
function lindblad_jump(Op::DiffEqLiouvillian{true,false}, u, p, t::Real)
    s = p(t)
    w, v = haml_eigs(Op.H, s, Op.lvl)
    gap_idx = GapIndices(w, Op.digits, Op.sigdigits)
    resample([ame_jump(x, u, gap_idx, v, s) for x in Op.opensys_eig], u)
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