Base.summary(H::AbstractHamiltonian) =
    string(
        TYPE_COLOR,
        nameof(typeof(H)),
        NO_COLOR,
        " with ",
        TYPE_COLOR,
        typeof(H).parameters[1],
        NO_COLOR
    )

function Base.show(io::IO, A::AbstractHamiltonian)
    println(io, summary(A))
    print(io, "with size: ")
    show(io, A.size)
end

"""
    evaluate(H::AbstractHamiltonian, s)

Evaluate the time dependent Hamiltonian at time s with the unit of `GHz`
"""
function evaluate(H::AbstractHamiltonian, s)
    H.(s) / 2 / π
end

"""
    function to_dense(H::SparseHamiltonian)

Convert SparseHamiltonian to DenseHamiltonian.
"""
function to_dense(H::SparseHamiltonian)
    m_dense = [Array(x) for x in H.m]
    DenseHamiltonian(H.f, m_dense, Array(H.u_cache), H.size)
end

"""
    function to_sparse(H::DenseHamiltonian)

Convert DenseHamiltonian to SparseHamiltonian.
"""
function to_sparse(H::DenseHamiltonian)
    m_sparse = [sparse(x) for x in H.m]
    cache = sum(m_sparse.m)
    fill!(cache, 0.0 + 0.0im)
    SparseHamiltonian(H.f, m_sparse, cache, H.size)
end

"""
    function to_real(H::AbstractHamiltonian{T}) where T<:Complex

Convert a Hamiltonian with Complex type to a Hamiltonian with real type. The converted Hamiltonian is used for projection to low-level subspaces and should not be used for any ODE calculations.
"""
function to_real(H::DenseHamiltonian{T}) where T<:Complex
    m_real = [real(x) for x in H.m]
    DenseHamiltonian(H.f, m_real)
end

function to_real(H::SparseHamiltonian{T}) where T<:Complex
    m_real = [real(x) for x in H.m]
    SparseHamiltonian(H.f, m_real)
end
