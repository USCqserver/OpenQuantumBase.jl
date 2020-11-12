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

struct Lindblad <: AbstractInteraction
    """Lindblad rate"""
    γ::Any
    """Lindblad operator"""
    L::Any
    """size"""
    size::Tuple
end

Lindblad(γ::Number, L::Matrix) = Lindblad((s) -> γ, (s) -> L, size(L))
Lindblad(γ::Number, L) = Lindblad((s) -> γ, L, size(L(0)))
Lindblad(γ, L::Matrix) = Lindblad(γ, (s) -> L, size(L))

function Lindblad(γ, L)
    if !(typeof(γ(0)) <: Number)
        throw(ArgumentError("γ should return a number."))
    end
    if !(typeof(L(0)) <: Matrix)
        throw(ArgumentError("L should return a matrix."))
    end
    Lindblad(γ, L, size(L(0)))
end

Base.size(lind::Lindblad) = lind.size

struct LindbladSet <: AbstractLiouvillian
    """1-d array of Lindblad rates"""
    γ::Vector
    """1-d array of Lindblad operataors"""
    L::Vector
    """size"""
    size::Tuple
end

Base.length(lind::LindbladSet) = length(lind.γ)

function LindbladSet(L::Vector{Lindblad})
    if any((x) -> size(x) != size(L[1]), L)
        throw(ArgumentError("All Lindblad operators should have the same size."))
    end
    LindbladSet([lind.γ for lind in L], [lind.L for lind in L], size(L[1]))
end

function (Lind::LindbladSet)(du, u, p, t)
    s = p(t)
    for (γfun, Lfun) in zip(Lind.γ, Lind.L)
        L = Lfun(s)
        γ = γfun(s)
        du .+= γ * (L * u * L' - 0.5 * ( L' * L * u + u * L' * L))
    end
end

function update_cache!(cache, lind::LindbladSet, p, t::Real)
    s = p(t)
    for (γfun, Lfun) in zip(lind.γ, lind.L)
        L = Lfun(s)
        γ = γfun(s)
        cache .-= 0.5 * γ * L' * L
    end
end

function lind_jump(lind::LindbladSet, u, p, t::Real)
    s = p(t)
    l = length(lind)
    prob = Float64[]
    ops = Vector{Matrix{ComplexF64}}()
    for (γfun, Lfun) in zip(lind.γ, lind.L)
        L = Lfun(s)
        γ = γfun(s)
        push!(prob, γ * norm(L * u))
        push!(ops, L)
    end
    sample(ops, Weights(prob))
end

lind_jump(Op::OpenSysOp{false,false}, u, p, t::Real) = lind_jump(Op.opensys[1], u, p, t)