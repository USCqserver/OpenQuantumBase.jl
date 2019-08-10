Base.summary(H::AbstractHamiltonian) = string(TYPE_COLOR, nameof(typeof(H)),
                                       NO_COLOR, " with ",
                                       TYPE_COLOR, typeof(H).parameters[1],
                                       NO_COLOR)

function Base.show(io::IO, A::AbstractHamiltonian)
    println(io, summary(A))
    print(io, "with size: ")
    show(io, A.size)
end

"""
    evaluate(H::AbstractHamiltonian, t)

Evaluate the time dependent Hamiltonian at time t with the unit of `GHz`
"""
function evaluate(H::AbstractHamiltonian, t)
    H.(t)/2/π
end

function to_dense(H::SparseHamiltonian)
    m_dense = [Array(x) for x in H.m]
    DenseHamiltonian(H.f, m_dense, Array(H.u_cache), H.size)
end

function to_sparse(H::DenseHamiltonian)
    m_sparse = [sparse(x) for x in H.m]
    cache = sum(m_sparse.m)
    fill!(cache, 0.0+0.0im)
    SparseHamiltonian(H.f, m_sparse, cache, H.size)
end
