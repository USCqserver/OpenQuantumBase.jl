"""
$(TYPEDEF)

Defines Redfield operator

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
end


Redfield(ops::ConstantCouplings, unitary, cfun) =
    Redfield{true}(ops, unitary, cfun)

Redfield(ops::AbstractTimeDependentCouplings, unitary, cfun) =
    Redfield{false}(ops, unitary, cfun)


function (R::Redfield{true})(du, u, tf::Real, t::Real)
    tf² = tf^2
    for S in R.ops
        #TODO Expose the error tolerance for integration
        function integrand(x)
            unitary = R.unitary(t) * R.unitary(x)'
            R.cfun(t - x) * unitary * S * unitary'
        end
        Λ, err = quadgk(integrand, 0, t, rtol = 1e-6, atol = 1e-8)
        𝐊₂ = S * Λ * u - Λ * u * S
        𝐊₂ = 𝐊₂ + 𝐊₂'
        axpy!(-tf², 𝐊₂, du)
    end
end


(R::Redfield{true})(du, u, tf::UnitTime, t::Real) = R(du, u, 1.0, t)



function (R::Redfield{false})(du, u, tf::Real, t::Real)
    tf² = tf^2
    for S in R.ops
        function integrand(x)
            unitary = R.unitary(t) * R.unitary(x)'
            R.cfun(t - x) * unitary * S(x) * unitary'
        end
        Λ, err = quadgk(integrand, 0, t, rtol = 1e-6, atol = 1e-8)
        𝐊₂ = S(t) * Λ * u - Λ * u * S(t)
        𝐊₂ = 𝐊₂ + 𝐊₂'
        axpy!(-tf², 𝐊₂, du)
    end
end


function (R::Redfield{false})(du, u, tf::UnitTime, t::Real)
    for S in R.ops
        function integrand(x)
            unitary = R.unitary(t) * R.unitary(x)'
            R.cfun(t - x) * unitary * S(x / tf) * unitary'
        end
        Λ, err = quadgk(integrand, 0, t, rtol = 1e-6, atol = 1e-8)
        𝐊₂ = S(t / tf) * Λ * u - Λ * u * S(t / tf)
        𝐊₂ = 𝐊₂ + 𝐊₂'
        axpy!(-1.0, 𝐊₂, du)
    end
end


function update_vectorized_cache!(cache, R::Redfield{true}, tf::Real, t::Real)
    tf² = tf^2
    iden = Matrix{eltype(cache)}(I, size(R.ops))
    for S in R.ops
        function integrand(x)
            unitary = R.unitary(t) * R.unitary(x)'
            R.cfun(t - x) * unitary * S * unitary'
        end
        Λ, err = quadgk(integrand, 0, t, rtol = 1e-6, atol = 1e-8)
        SΛ = S * Λ
        cache .-=
            tf² * (iden ⊗ SΛ + conj(SΛ) ⊗ iden - transpose(S) ⊗ Λ - conj(Λ) ⊗ S)
    end
end


update_vectorized_cache!(cache, R::Redfield{true}, tf::UnitTime, t::Real) =
    update_vectorized_cache!(cache, R, 1.0, t)


function update_vectorized_cache!(cache, R::Redfield{false}, tf::Real, t::Real)
    tf² = tf^2
    iden = Matrix{eltype(cache)}(I, size(R.ops))
    for S in R.ops
        function integrand(x)
            u = R.unitary(t) * R.unitary(x)'
            R.cfun(t - x) * u * S(x) * u'
        end
        Λ, err = quadgk(integrand, 0, t, rtol = 1e-6, atol = 1e-8)
        Sm = S(t)
        SΛ = Sm * Λ
        cache .-=
            tf² *
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
        Λ, err = quadgk(integrand, 0, t, rtol = 1e-6, atol = 1e-8)
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
