struct ULindblad <: AbstractLiouvillian
    """Lindblad kernels"""
    kernels::Any
    """close system unitary"""
    unitary::Any
    """absolute error tolerance for integration"""
    atol::Float64
    """relative error tolerance for integration"""
    rtol::Float64
    """cache matrix for inplace unitary"""
    Ut::Union{Matrix,MMatrix}
    """cache matrix for inplace unitary"""
    Uτ::Union{Matrix,MMatrix}
    """cache matrix for integration"""
    Λ::Union{Matrix,MMatrix}
    """cache matrix for Lindblad operator"""
    LO::Union{Matrix,MMatrix}
    """tf minus coarse grain time scale"""
    Ta::Real
end

function ULindblad(kernels, U, Ta, atol, rtol)
    m_size = size(kernels[1][2])
    Λ = m_size[1] <= 10 ? zeros(MMatrix{m_size[1],m_size[2],ComplexF64}) :
        zeros(ComplexF64, m_size[1], m_size[2])

    unitary = isinplace(U) ? U.func : (cache, t) -> cache .= U(t)
    ULindblad(kernels, unitary, atol, rtol, similar(Λ),
        similar(Λ), similar(Λ), Λ, Ta)
end

function (L::ULindblad)(du, u, p, t)
    s = p(t)
    LO = fill!(L.LO, 0.0)
    for (inds, coupling, cfun) in L.kernels
        for (i, j) in inds
            function integrand(cache, x)
                L.unitary(L.Ut, t)
                L.unitary(L.Uτ, x)
                L.Ut .= L.Ut * L.Uτ'
                mul!(L.Uτ, coupling[j](s), L.Ut')
                mul!(cache, L.Ut, L.Uτ, cfun[i, j](t, x), 0)
            end
            quadgk!(
                integrand,
                L.Λ,
                max(0.0, t - L.Ta),
                min(t + L.Ta, p.tf),
                rtol=L.rtol,
                atol=L.atol,
            )
            axpy!(1.0, L.Λ, LO)
        end
    end
    du .+= LO * u * LO' - 0.5 * (LO' * LO * u + u * LO' * LO)
end