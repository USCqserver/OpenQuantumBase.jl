"""
$(TYPEDEF)

Defines Redfield operator

# Fields

$(FIELDS)
"""
struct Redfield <: AbstractOpenSys
    """system-bath coupling operator"""
    ops
    """close system unitary"""
    unitary
    """bath correlation function"""
    cfun
end


function (R::Redfield)(du, u, tf::Real, t::Real)
    tf² = tf^2
    for S in R.ops
        Λ, err = Λ_calculation(t, S, R.cfun, R.unitary; rtol = 1e-6, atol = 1e-8)
        𝐊₂ = redfield_K(S, Λ, u, t)
        𝐊₂ = 𝐊₂ + 𝐊₂'
        axpy!(-tf², 𝐊₂, du)
    end
end


function (R::Redfield)(du, u, tf::UnitTime, t::Real)
    for S in R.ops
        Λ, err = Λ_calculation(t, S, R.cfun, R.unitary; rtol = 1e-6, atol = 1e-8)
        𝐊₂ = redfield_K(S, Λ, u, t)
        𝐊₂ = 𝐊₂ + 𝐊₂'
        axpy!(-1.0, 𝐊₂, du)
    end
end


@inline redfield_K(S::Matrix{T}, Λ, u, t) where {T<:Number} = S * Λ * u - Λ * u * S
@inline redfield_K(S, Λ, u, t) = S(t) * Λ * u - Λ * u * S(t)


function Λ_calculation(t, op::Matrix{T}, cfun, unitary; rtol = 1e-8, atol = 1e-8) where {T<:Number}
    function integrand(x)
        u = unitary(t) * unitary(x)'
        cfun(t - x) * u * op * u'
    end
    res = quadgk(integrand, 0, t, rtol = rtol, atol = atol)
end


function Λ_calculation(t, op, cfun, unitary; rtol = 1e-8, atol = 1e-8)
    function integrand(x)
        u = unitary(t) * unitary(x)'
        cfun(t - x) * u * op(x) * u'
    end
    res = quadgk(integrand, 0, t, rtol = rtol, atol = atol)
end


function update_vectorized_cache!(cache, R::Redfield, tf::Real, t::Real)
    tf² = tf^2
    iden = Matrix{eltype(cache)}(I, size(R.ops))
    for S in R.ops
        Λ, err = Λ_calculation(t, S, R.cfun, R.unitary; rtol = 1e-6, atol = 1e-8)
        SΛ = S * Λ
        cache .-= tf² * (iden ⊗ SΛ + conj(SΛ) ⊗ iden - transpose(S) ⊗ Λ - conj(Λ) ⊗ S)
    end
end
