import StaticArrays: MMatrix
import QuadGK: quadgk!

"""
$(TYPEDEF)

Defines Redfield operator.

# Fields

$(FIELDS)
"""
struct Redfield <: AbstractOpenSys
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
end

function Redfield(ops::AbstractCouplings, U, cfun; atol = 1e-8, rtol = 1e-6)
    m_size = size(ops)
    if m_size[1] <= 1
        Λ = zeros(MMatrix{m_size[1],m_size[2],ComplexF64})
    else
        Λ = zeros(ComplexF64, m_size[1], m_size[2])
    end
    if isinplace(U)
        unitary = U.func
    else
        unitary = (cache, t) -> cache .= U(t)
    end
    Redfield(ops, unitary, cfun, atol, rtol, similar(Λ), similar(Λ), Λ)
end

function (R::Redfield)(du, u, tf::Real, t::Real)
    for S in R.ops
        function integrand(cache, x)
            R.unitary(R.Ut, t)
            R.unitary(R.Uτ, x)
            R.Ut .= R.Ut * R.Uτ'
            mul!(R.Uτ, S(t), R.Ut')
            mul!(cache, R.Ut, R.Uτ, tf * R.cfun(t - x), 0)
        end
        quadgk!(integrand, R.Λ, 0.0, t, rtol = R.rtol, atol = R.atol)
        𝐊₂ = S(t) * R.Λ * u - R.Λ * u * S(t)
        𝐊₂ = 𝐊₂ + 𝐊₂'
        axpy!(-tf, 𝐊₂, du)
    end
end

function (R::Redfield)(du, u, tf::UnitTime, t::Real)
    for S in R.ops
        function integrand(cache, x)
            R.unitary(R.Ut, t)
            R.unitary(R.Uτ, x)
            R.Ut .= R.Ut * R.Uτ'
            mul!(R.Uτ, S(t / tf), R.Ut')
            mul!(cache, R.Ut, R.Uτ, R.cfun(t - x), 0)
        end
        quadgk!(integrand, R.Λ, 0.0, t, rtol = R.rtol, atol = R.atol)
        𝐊₂ = S(t / tf) * R.Λ * u - R.Λ * u * S(t / tf)
        𝐊₂ = 𝐊₂ + 𝐊₂'
        axpy!(-1.0, 𝐊₂, du)
    end
end

update_ρ!(du, u, p::ODEParams, t::Real, R::Redfield) = R(du, u, p.tf, t)

function update_vectorized_cache!(cache, R::Redfield, tf::Real, t::Real)
    iden = Matrix{eltype(cache)}(I, size(R.ops))
    for S in R.ops
        function integrand(cache, x)
            R.unitary(R.Ut, t)
            R.unitary(R.Uτ, x)
            R.Ut .= R.Ut * R.Uτ'
            mul!(R.Uτ, S(t), R.Ut')
            mul!(cache, R.Ut, R.Uτ, tf * R.cfun(t - x), 0)
        end
        quadgk!(integrand, R.Λ, 0.0, t, rtol = R.rtol, atol = R.atol)
        SS = S(t)
        SΛ = SS * R.Λ
        cache .-=
            tf *
            (iden ⊗ SΛ + conj(SΛ) ⊗ iden - transpose(SS) ⊗ R.Λ - conj(R.Λ) ⊗ SS)
    end
end

function update_vectorized_cache!(cache, R::Redfield, tf::UnitTime, t::Real)
    iden = Matrix{eltype(cache)}(I, size(R.ops))
    for S in R.ops
        function integrand(cache, x)
            R.unitary(R.Ut, t)
            R.unitary(R.Uτ, x)
            R.Ut .= R.Ut * R.Uτ'
            mul!(R.Uτ, S(t / tf), R.Ut')
            mul!(cache, R.Ut, R.Uτ, R.cfun(t - x), 0)
        end
        quadgk!(integrand, R.Λ, 0.0, t, rtol = R.rtol, atol = R.atol)
        SS = S(t / tf)
        SΛ = SS * R.Λ
        cache .-=
            (iden ⊗ SΛ + conj(SΛ) ⊗ iden - transpose(SS) ⊗ R.Λ - conj(R.Λ) ⊗ SS)
    end
end

update_vectorized_cache!(du, u, p::ODEParams, t::Real, R::Redfield) =
    update_vectorized_cache!(du, R, p.tf, t)

struct RedfieldSet{T<:Tuple} <: AbstractOpenSys
    """Redfield operators"""
    reds::T
end

RedfieldSet(red::Redfield...) = RedfieldSet(red)

function (R::RedfieldSet)(du, u, tf, t)
    for r in R.reds
        r(du, u, tf, t)
    end
end

function update_vectorized_cache!(cache, R::RedfieldSet, tf, t)
    for r in R.reds
        update_vectorized_cache!(cache, r, tf, t)
    end
end
