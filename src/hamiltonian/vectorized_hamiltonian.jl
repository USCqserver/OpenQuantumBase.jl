# vectorized hamiltonian implementation is still experimental
struct VectorizedHamiltonian{T<:Complex} <: AbstractHamiltonian{T}
    H::AbstractHamiltonian{T}
    size
end

function VectorizedHamiltonian(H)
    VectorizedHamiltonian(H, H.size)
end

function (H::VectorizedHamiltonian{T})(tf, t::Real) where T<:Complex
    hmat = H.H(tf, t)
    iden = Matrix{T}(I, H.H.size)
    res = -1.0im * (iden⊗hmat - transpose(hmat)⊗iden)
end
