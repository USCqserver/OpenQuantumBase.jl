"""
$(TYPEDEF)

Defines a fluctuator ensemble controller

# Fields

$(FIELDS)
"""
mutable struct Fluctuators <: AbstractLiouvillian
    """system-bath coupling operator"""
    coupling::Any
    """waitting time distribution for every fluctuators"""
    dist::Any
    """cache for each fluctuator value"""
    b0::Any
    """index of the fluctuator to be flipped next"""
    next_idx::Any
    """time interval for next flip event"""
    next_τ::Any
    """noise value"""
    n::Any
end

function (F::Fluctuators)(du, u, p, t)
    s = p(t)
    H = sum(F.n .* F.coupling(s))
    du .= -1.0im * (H*u - u*H)
    # gemm! does not work for static matrix
    #gemm!('N', 'N', -1.0im, H, u, 1.0 + 0.0im, du)
    #gemm!('N', 'N', 1.0im, u, H, 1.0 + 0.0im, du)
end

function update_cache!(cache, F::Fluctuators, p, t)
    s = p(t)
    cache .+= -1.0im * sum(F.n .* F.coupling(s))
end

function update_vectorized_cache!(cache, F::Fluctuators, p, t)
    s = p(t)
    hmat = sum(F.n .* F.coupling(s))
    iden = one(hmat)
    cache .+= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end

function next_state!(F::Fluctuators)
    next_τ, next_idx = findmin(rand(F.dist, size(F.b0, 2)))
    F.next_τ = next_τ
    F.next_idx = next_idx
    F.b0[next_idx] *= -1
    nothing
end

function reset!(F::Fluctuators, initializer)
    F.b0 = abs.(F.b0) .* initializer(length(F.dist), size(F.b0, 2))
    F.n = sum(F.b0, dims = 1)[:]
    next_state!(F)
end

FluctuatorOperator(H, flist) = OpenSysOp(H, flist, size(H,1))
