import HCubature: hcubature

"""
$(TYPEDEF)

Defines CGME operator.

# Fields

$(FIELDS)
"""
struct CGOP{is_const} <: AbstractOpenSys
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

CGOP(ops::ConstantCouplings, unitary, cfun, Ta; atol = 1e-8, rtol = 1e-6) =
    CGOP{true}(ops, unitary, cfun, Ta, atol, rtol)

CGOP(
    ops::AbstractTimeDependentCouplings,
    unitary,
    cfun,
    Ta;
    atol = 1e-8,
    rtol = 1e-6,
) = CGOP{false}(ops, unitary, cfun, Ta, atol, rtol)

function (CG::CGOP{true})(du, u, tf::Real, t::Real)
    # in this case, time in physical unit is t * tf
    # Ta is also in physical unit
    lower_bound = t * tf < CG.Ta / 2 ? [-t, -t] : [-CG.Ta, -CG.Ta] / tf / 2
    upper_bound =
        t * tf + CG.Ta / 2 > tf ? [1.0, 1.0] .- t : [CG.Ta, CG.Ta] / tf / 2
    Ut = CG.unitary(t)
    for S in CG.ops
        function integrand(x)
            Ut2 = CG.unitary(t + x[2]) * Ut'
            Ut1 = CG.unitary(t + x[1]) * Ut'
            At1 = Ut1' * S * Ut1
            At2 = Ut2' * S * Ut2
            tf^2 *
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
        axpy!(tf, cg_res, du)
    end
end

(R::CGOP{true})(du, u, tf::UnitTime, t::Real) = R(du, u, 1.0, t)

function (CG::CGOP{false})(du, u, tf::Real, t::Real)
    # in this case, time in physical unit is t * tf
    # Ta is also in physical unit
    lower_bound = t * tf < CG.Ta / 2 ? [-t, -t] : [-CG.Ta, -CG.Ta] / tf / 2
    upper_bound =
        t * tf + CG.Ta / 2 > tf ? [1.0, 1.0] .- t : [CG.Ta, CG.Ta] / tf / 2
    Ut = CG.unitary(t)
    for S in CG.ops
        function integrand(x)
            Ut2 = CG.unitary(t + x[2]) / Ut
            Ut1 = CG.unitary(t + x[1]) / Ut
            At1 = Ut1' * S(t + x[1]) * Ut1
            At2 = Ut2' * S(t + x[2]) * Ut2
            tf^2 *
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
        axpy!(tf, cg_res, du)
    end
end

function (CG::CGOP{false})(du, u, tf::UnitTime, t::Real)
    # in this case, time in physical unit is t * tf
    # Ta is also in physical unit
    lower_bound = t * tf < CG.Ta / 2 ? [-t, -t] : [-CG.Ta, -CG.Ta] / 2
    upper_bound =
        t * tf + CG.Ta / 2 > tf ? [1.0, 1.0] .- t : [CG.Ta, CG.Ta] / 2
    Ut = CG.unitary(t)
    for S in CG.ops
        function integrand(x)
            Ut2 = CG.unitary(t + x[2]) / Ut
            Ut1 = CG.unitary(t + x[1]) / Ut
            At1 = Ut1' * S(t / tf + x[1]) * Ut1
            At2 = Ut2' * S(t / tf + x[2]) * Ut2
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
