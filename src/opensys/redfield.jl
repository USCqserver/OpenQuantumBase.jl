struct Redfield <: AbstractOpenSys
    ops
    unitary
    cfun
end

function (R::Redfield)(du, u, p, t)
    tf² = p.tf^2
    for S in R.ops
        Λ, err = Λ_calculation(t, S, R.cfun, R.unitary; rtol=1e-6, atol=1e-8)
        𝐊₂ = S*Λ*u - Λ*u*S
        𝐊₂ = 𝐊₂ + 𝐊₂'
        axpy!(-tf², 𝐊₂, du)
    end
end

function Λ_calculation(t, op::Matrix{T}, cfun, unitary; rtol=1e-8, atol=1e-8) where T
    function integrand(x)
        u = unitary(t)*unitary(x)'
        cfun(t-x) * u * op * u'
    end
    res = quadgk(integrand, 0, t, rtol=rtol, atol=atol)
end
