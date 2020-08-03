import HCubature: hcubature

"""
$(TYPEDEF)

Defines CGME operator.

# Fields

$(FIELDS)
"""
struct CGGenerator <: AbstractLiouvillian
    """system-bath coupling operator"""
    ops::Any
    """close system unitary"""
    unitary::Any
    """bath correlation function"""
    cfun::Any
    """coarse grain timescale"""
    Ta::Any
    """absolute error tolerance for integration"""
    atol::Float64
    """relative error tolerance for integration"""
    rtol::Float64
end

CGGenerator(ops::AbstractCouplings, unitary, cfun, Ta; atol = 1e-8, rtol = 1e-6) =
    CGGenerator(ops, unitary, cfun, Ta, atol, rtol)

function (CG::CGGenerator)(du, u, p, t::Real)
    tf = p.tf
    # in this case, time in physical unit is t * tf
    # Ta is also in physical unit
    lower_bound = t < CG.Ta / 2 ? [-t, -t] : [-CG.Ta, -CG.Ta] / 2
    upper_bound = t + CG.Ta / 2 > tf ? [tf, tf] .- t : [CG.Ta, CG.Ta] / 2
    Ut = CG.unitary(t)
    for S in CG.ops
        function integrand(x)
            Ut2 = CG.unitary(t + x[2]) / Ut
            Ut1 = CG.unitary(t + x[1]) / Ut
            At1 = Ut1' * S(p(t + x[1])) * Ut1
            At2 = Ut2' * S(p(t + x[2])) * Ut2
            CG.cfun(x[2] - x[1]) *
            (At1 * u * At2 - 0.5 * (At2 * At1 * u + u * At2 * At1)) / CG.Ta
        end
        cg_res, err = hcubature(
            integrand,
            lower_bound,
            upper_bound,
            rtol = CG.rtol,
            atol = CG.atol,
        )
        axpy!(1.0, cg_res, du)
    end
end
