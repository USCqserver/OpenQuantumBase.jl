"""
$(TYPEDEF)

Defines Redfield operator.

# Fields

$(FIELDS)
"""
struct Redfield{is_const} <: AbstractOpenSys
    """system-bath coupling operator"""
    ops
    """close system unitary"""
    unitary
    """bath correlation function"""
    cfun
    """absolute error tolerance for integration"""
    atol::Float64
    """relative error tolerance for integration"""
    rtol::Float64
end


Redfield(ops::ConstantCouplings, unitary, cfun; atol = 1e-8, rtol = 1e-6) =
    Redfield{true}(ops, unitary, cfun, atol, rtol)

Redfield(
    ops::AbstractTimeDependentCouplings,
    unitary,
    cfun;
    atol = 1e-8,
    rtol = 1e-6,
) = Redfield{false}(ops, unitary, cfun, atol, rtol)


function (R::Redfield{true})(du, u, tf::Real, t::Real)
    for S in R.ops
        function integrand(x)
            unitary = R.unitary(t) * R.unitary(x)'
            tf * R.cfun(t - x) * unitary * S * unitary'
        end
        Λ, err = quadgk(integrand, 0, t, rtol = R.rtol, atol = R.atol)
        𝐊₂ = S * Λ * u - Λ * u * S
        𝐊₂ = 𝐊₂ + 𝐊₂'
        axpy!(-tf, 𝐊₂, du)
    end
end


(R::Redfield{true})(du, u, tf::UnitTime, t::Real) = R(du, u, 1.0, t)



function (R::Redfield{false})(du, u, tf::Real, t::Real)
    for S in R.ops
        function integrand(x)
            unitary = R.unitary(t) * R.unitary(x)'
            tf * R.cfun(t - x) * unitary * S(x) * unitary'
        end
        Λ, err = quadgk(integrand, 0, t, rtol = R.rtol, atol = R.atol)
        𝐊₂ = S(t) * Λ * u - Λ * u * S(t)
        𝐊₂ = 𝐊₂ + 𝐊₂'
        axpy!(-tf, 𝐊₂, du)
    end
end


function (R::Redfield{false})(du, u, tf::UnitTime, t::Real)
    for S in R.ops
        function integrand(x)
            unitary = R.unitary(t) * R.unitary(x)'
            R.cfun(t - x) * unitary * S(x / tf) * unitary'
        end
        Λ, err = quadgk(integrand, 0, t, rtol = R.rtol, atol = R.atol)
        𝐊₂ = S(t / tf) * Λ * u - Λ * u * S(t / tf)
        𝐊₂ = 𝐊₂ + 𝐊₂'
        axpy!(-1.0, 𝐊₂, du)
    end
end


function update_vectorized_cache!(cache, R::Redfield{true}, tf::Real, t::Real)
    iden = Matrix{eltype(cache)}(I, size(R.ops))
    for S in R.ops
        function integrand(x)
            unitary = R.unitary(t) * R.unitary(x)'
            tf * R.cfun(t - x) * unitary * S * unitary'
        end
        Λ, err = quadgk(integrand, 0, t, rtol = R.rtol, atol = R.atol)
        SΛ = S * Λ
        cache .-=
            tf * (iden ⊗ SΛ + conj(SΛ) ⊗ iden - transpose(S) ⊗ Λ - conj(Λ) ⊗ S)
    end
end


update_vectorized_cache!(cache, R::Redfield{true}, tf::UnitTime, t::Real) =
    update_vectorized_cache!(cache, R, 1.0, t)


function update_vectorized_cache!(cache, R::Redfield{false}, tf::Real, t::Real)
    iden = Matrix{eltype(cache)}(I, size(R.ops))
    for S in R.ops
        function integrand(x)
            u = R.unitary(t) * R.unitary(x)'
            tf * R.cfun(t - x) * u * S(x) * u'
        end
        Λ, err = quadgk(integrand, 0, t, rtol = R.rtol, atol = R.atol)
        Sm = S(t)
        SΛ = Sm * Λ
        cache .-=
            tf *
            (iden ⊗ SΛ + conj(SΛ) ⊗ iden - transpose(Sm) ⊗ Λ - conj(Λ) ⊗ Sm)
    end
end


function update_vectorized_cache!(
    cache,
    R::Redfield{false},
    tf::UnitTime,
    t::Real,
)
    iden = Matrix{eltype(cache)}(I, size(R.ops))
    for S in R.ops
        function integrand(x)
            u = R.unitary(t) * R.unitary(x)'
            R.cfun(t - x) * u * S(x / tf) * u'
        end
        Λ, err = quadgk(integrand, 0, t, rtol = R.rtol, atol = R.atol)
        Sm = S(t / tf)
        SΛ = Sm * Λ
        cache .-= iden ⊗ SΛ + conj(SΛ) ⊗ iden - transpose(Sm) ⊗ Λ - conj(Λ) ⊗ Sm
    end
end


struct RedfieldSet{T<:Tuple} <: AbstractOpenSys
    """Redfield operators"""
    reds::T
end


function RedfieldSet(red::Redfield...)
    RedfieldSet(red)
end


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
