import HCubature: hcubature

"""
$(TYPEDEF)

`CGLiouvillian` defines the Liouvillian operator corresponding to the CGME.

# Fields

$(FIELDS)
"""
struct CGLiouvillian <: AbstractLiouvillian
    "CGME kernels"
    kernels::Any
    "close system unitary"
    unitary::Any
    "absolute error tolerance for integration"
    atol::Float64
    "relative error tolerance for integration"
    rtol::Float64
    "cache matrix for inplace unitary"
    Ut1::Union{Matrix,MMatrix}
    "cache matrix for inplace unitary"
    Ut2::Union{Matrix,MMatrix}
    "cache matrix for integration"
    Ut::Union{Matrix,MMatrix}
end

function CGLiouvillian(kernels, U, atol, rtol)
    m_size = size(kernels[1][2])
    Λ = m_size[1] <= 10 ? zeros(MMatrix{m_size[1],m_size[2],ComplexF64}) :
        zeros(ComplexF64, m_size[1], m_size[2])
    unitary = isinplace(U) ? U.func : (cache, t) -> cache .= U(t)
    CGLiouvillian(kernels, unitary, atol, rtol, similar(Λ), similar(Λ), Λ)
end

function (CG::CGLiouvillian)(du, u, p, t::Real)
    tf = p.tf
    for (inds, coupling, cfun, Ta) in CG.kernels
        lower_bound = t < Ta / 2 ? [-t, -t] : [-Ta, -Ta] / 2
        upper_bound = t + Ta / 2 > tf ? [tf, tf] .- t : [Ta, Ta] / 2
        for (i, j) in inds
            function integrand(x)
                Ut = CG.Ut
                Ut1 = CG.Ut1
                Ut2 = CG.Ut2
                CG.unitary(Ut, t)
                Ut2 .= CG.unitary(Ut2, t + x[2]) * Ut'
                Ut1 .= CG.unitary(Ut1, t + x[1]) * Ut'
                At1 = Ut .= Ut1' * coupling[i](p(t + x[1])) * Ut1
                At2 = Ut1 .= Ut2' * coupling[j](p(t + x[2])) * Ut2
                cfun[i, j](t + x[2], t + x[1]) *
                (At1 * u * At2 - 0.5 * (At2 * At1 * u + u * At2 * At1)) / Ta
            end
            cg_res, err = hcubature(
                integrand,
                lower_bound,
                upper_bound,
                rtol=CG.rtol,
                atol=CG.atol,
            )
            axpy!(1.0, cg_res, du)
        end
    end
end
