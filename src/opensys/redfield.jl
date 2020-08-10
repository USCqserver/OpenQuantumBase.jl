import StaticArrays: MMatrix
import QuadGK: quadgk!

abstract type AbstractRedfield <: AbstractLiouvillian end

"""
$(TYPEDEF)

Defines DiagRedfieldGenerator.

# Fields

$(FIELDS)
"""
struct DiagRedfieldGenerator <: AbstractRedfield
    """system-bath coupling operator"""
    ops::AbstractCouplings
    """close system unitary"""
    unitary::Any
    """bath correlation function"""
    cfun::Any
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
    """tf minus coarse grain time scale"""
    Ta::Number
end

function DiagRedfieldGenerator(
    ops::AbstractCouplings,
    U,
    cfun,
    Ta;
    atol = 1e-8,
    rtol = 1e-6,
)
    m_size = size(ops)
    if m_size[1] <= 10
        Λ = zeros(MMatrix{m_size[1],m_size[2],ComplexF64})
    else
        Λ = zeros(ComplexF64, m_size[1], m_size[2])
    end
    if isinplace(U)
        unitary = U.func
    else
        unitary = (cache, t) -> cache .= U(t)
    end
    DiagRedfieldGenerator(
        ops,
        unitary,
        cfun,
        atol,
        rtol,
        similar(Λ),
        similar(Λ),
        Λ,
        Ta,
    )
end

function (R::DiagRedfieldGenerator)(du, u, p, t::Real)
    s = p(t)
    for S in R.ops
        function integrand(cache, x)
            R.unitary(R.Ut, t)
            R.unitary(R.Uτ, x)
            R.Ut .= R.Ut * R.Uτ'
            mul!(R.Uτ, S(s), R.Ut')
            mul!(cache, R.Ut, R.Uτ, R.cfun(t - x), 0)
        end
        quadgk!(
            integrand,
            R.Λ,
            max(0.0, t - R.Ta),
            t,
            rtol = R.rtol,
            atol = R.atol,
        )
        𝐊₂ = S(s) * R.Λ * u - R.Λ * u * S(s)
        𝐊₂ = 𝐊₂ + 𝐊₂'
        axpy!(-1.0, 𝐊₂, du)
    end
end

function update_vectorized_cache!(cache, R::DiagRedfieldGenerator, p, t::Real)
    iden = Matrix{eltype(cache)}(I, size(R.ops))
    s = p(t)
    for S in R.ops
        function integrand(cache, x)
            R.unitary(R.Ut, t)
            R.unitary(R.Uτ, x)
            R.Ut .= R.Ut * R.Uτ'
            mul!(R.Uτ, S(s), R.Ut')
            mul!(cache, R.Ut, R.Uτ, R.cfun(t - x), 0)
        end
        quadgk!(
            integrand,
            R.Λ,
            max(0.0, t - R.Ta),
            t,
            rtol = R.rtol,
            atol = R.atol,
        )
        SS = S(s)
        SΛ = SS * R.Λ
        cache .-=
            (iden ⊗ SΛ + conj(SΛ) ⊗ iden - transpose(SS) ⊗ R.Λ - conj(R.Λ) ⊗ SS)
    end
end

RedfieldOperator(H, R) = OpenSysOp(H, R, size(H, 1))

"""
$(TYPEDEF)

Defines BaseRedfieldGenerator.

# Fields

$(FIELDS)
"""
struct BaseRedfieldGenerator <: AbstractRedfield
    """system-bath coupling operator"""
    ops::AbstractCouplings
    """close system unitary"""
    unitary::Any
    """bath correlation function"""
    cfun::Any
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
    """tf minus coarse grain time scale"""
    Ta::Number
end