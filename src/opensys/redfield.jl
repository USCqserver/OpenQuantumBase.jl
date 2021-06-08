RedfieldOperator(H, R) = OpenSysOp(H, R, size(H, 1))

"""
$(TYPEDEF)

Defines RedfieldLiouvillian.

# Fields

$(FIELDS)
"""
struct RedfieldLiouvillian <: AbstractLiouvillian
    "Redfield kernels"
    kernels::Any
    "close system unitary"
    unitary::Any
    "absolute error tolerance for integration"
    atol::Float64
    "relative error tolerance for integration"
    rtol::Float64
    "cache matrix for inplace unitary"
    Ut::Union{Matrix,MMatrix}
    "cache matrix for inplace unitary"
    Uτ::Union{Matrix,MMatrix}
    "cache matrix for integration"
    Λ::Union{Matrix,MMatrix}
    "tf minus coarse grain time scale"
    Ta::Real
end

function RedfieldLiouvillian(kernels, U, Ta, atol, rtol)
    m_size = size(kernels[1][2])
    Λ = m_size[1] <= 10 ? zeros(MMatrix{m_size[1],m_size[2],ComplexF64}) :
        zeros(ComplexF64, m_size[1], m_size[2])
    # if the unitary does not in place operation, assign a pesudo inplace
    # function
    unitary = isinplace(U) ? U.func : (cache, t) -> cache .= U(t)
    RedfieldLiouvillian(kernels, unitary, atol, rtol, similar(Λ),
        similar(Λ), Λ, Ta)
end

function (R::RedfieldLiouvillian)(du, u, p, t::Real)
    s = p(t)
    for (inds, coupling, cfun) in R.kernels
        for (i, j) in inds
            function integrand(cache, x)
                R.unitary(R.Ut, t)
                R.unitary(R.Uτ, x)
                R.Ut .= R.Ut * R.Uτ'
                mul!(R.Uτ, coupling[j](p(x)), R.Ut')
                # The 5 arguments mul! will to produce NaN when it
                # should not. May switch back to it when this is fixed.
                # mul!(cache, R.Ut, R.Uτ, cfun[i, j](t, x), 0)
                mul!(cache, R.Ut, R.Uτ)
                lmul!(cfun[i, j](t, x), cache)
            end
            quadgk!(
                integrand,
                R.Λ,
                max(0.0, t - R.Ta),
                t,
                rtol=R.rtol,
                atol=R.atol,
            )
            SS = coupling[i](s)
            𝐊₂ = SS * R.Λ * u - R.Λ * u * SS
            𝐊₂ = 𝐊₂ + 𝐊₂'
            axpy!(-1.0, 𝐊₂, du)
        end
    end
end

function update_vectorized_cache!(cache, R::RedfieldLiouvillian, p, t::Real)
    iden = one(R.Λ)
    s = p(t)
    for (inds, coupling, cfun) in R.kernels
        for (i, j) in inds
            function integrand(cache, x)
                R.unitary(R.Ut, t)
                R.unitary(R.Uτ, x)
                R.Ut .= R.Ut * R.Uτ'
                mul!(R.Uτ, coupling[j](p(x)), R.Ut')
                # The 5 arguments mul! will to produce NaN when it
                # should not. May switch back to it when this is fixed.
                # mul!(cache, R.Ut, R.Uτ, cfun[i, j](t, x), 0)
                mul!(cache, R.Ut, R.Uτ)
                lmul!(cfun[i, j](t, x), cache)
            end
            quadgk!(
                integrand,
                R.Λ,
                max(0.0, t - R.Ta),
                t,
                rtol=R.rtol,
                atol=R.atol,
            )
            SS = coupling[i](s)
            SΛ = SS * R.Λ
            cache .-= (
                iden ⊗ SΛ + conj(SΛ) ⊗ iden - transpose(SS) ⊗ R.Λ -
                conj(R.Λ) ⊗ SS
            )
        end
    end
end