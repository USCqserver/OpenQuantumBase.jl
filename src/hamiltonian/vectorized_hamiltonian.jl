struct VectorizedHamiltonian
    H
end

function (H::VectorizedHamiltonian)(du, u::Matrix{T}, p::Real, t::Real) where T<:Complex
    hamil = H.H(t)
end
