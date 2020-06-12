Base.summary(H::AbstractHamiltonian) = string(
    TYPE_COLOR,
    nameof(typeof(H)),
    NO_COLOR,
    " with ",
    TYPE_COLOR,
    typeof(H).parameters[1],
    NO_COLOR,
)

function Base.show(io::IO, A::AbstractHamiltonian)
    println(io, summary(A))
    print(io, "with size: ")
    show(io, size(A))
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
function to_real(H::DenseHamiltonian{T}) where {T<:Complex}
    m_real = [real(x) for x in H.m]
    DenseHamiltonian(H.f, m_real)
end


function Base.real(H::AbstractHamiltonian{T}) where {T}
    S = real(T)
    convert(S, H)
end


function is_complex(f_list, m_list)
    any(m_list) do m
        eltype(m) <: Complex
    end || any(f_list) do f
        typeof(f(0)) <: Complex
    end
end


"""
    function eigen_decomp(H::AbstractHamiltonian, s; lvl::Int = 2, eig_init = EIGEN_DEFAULT) -> (w, v)

Calculate the eigen value decomposition of the Hamiltonian `H` at time `s`. Keyword argument `lvl` specifies the number of levels to keep in the output. `w` is a vector of eigenvalues and `v` is a matrix of the eigenvectors in the columns. (The `k`th eigenvector can be obtained from the slice `v[:, k]`.) `w` will be in unit of `GHz`.

`eig_init` is the initializer for eigen factorization routine. It returns a function of signature: `(H, s, lvl) -> (w, v)`. The default initializer `EIGEN_DEFAULT` will use `LAPACK` routine for both dense and sparse matrices.
"""
function eigen_decomp(
    H::AbstractHamiltonian,
    s;
    lvl::Int = 2,
    eig_init = EIGEN_DEFAULT,
)
    _eigs = eig_init(H)
    w, v = _eigs(H, s, lvl)
    lmul!(1 / 2 / π, real(w)), v
end


"""
    eigen_decomp(H::AbstractHamiltonian, s::AbstractArray{Float64,1}; lvl::Int = 2, eig_init = EIGEN_DEFAULT) -> (vals, vecs)

Calculate the eigen value decomposition of the Hamiltonian `H` at an array of time points `s`. The output keeps the lowest `lvl` eigenstates and their corresponding eigenvalues. Output `(vals, vecs)` have the dimensions of `(lvl, length(s))` and `(size(H, 1), lvl, length(s))` respectively.
"""
function eigen_decomp(
    H::AbstractHamiltonian,
    s::AbstractArray{Float64,1};
    lvl::Int = 2,
    eig_init = EIGEN_DEFAULT,
)
    s_dim = length(s)
    _eigs = eig_init(H)
    res_val = Array{eltype(H),2}(undef, (lvl, s_dim))
    res_vec = Array{eltype(H),3}(undef, (size(H, 1), lvl, s_dim))
    for (i, s_val) in enumerate(s)
        val, vec = _eigs(H, s_val, lvl)
        res_val[:, i] = val[1:lvl]
        res_vec[:, :, i] = vec[:, 1:lvl]
    end
    res_val, res_vec
end


"""
    function inst_population(s, states, H::AbstractHamiltonian; lvl=1:1, eig_init = EIGEN_DEFAULT)

For a time series quantum states given by `states`, whose time points are given by `s`, calculate the population of instantaneous eigenstates of `H`. The levels of the instantaneous eigenstates are specified by `lvl`, which can be any slice index.

`eig_init` is the initializer for eigen factorization routine. It returns a function of signature: `(H, s, lvl) -> (w, v)`. The default initializer `EIGEN_DEFAULT` will use `LAPACK` routine for both dense and sparse matrices.
"""
function inst_population(
    s,
    states,
    H::AbstractHamiltonian;
    lvl = 1:1,
    eig_init = EIGEN_DEFAULT,
)
    if typeof(lvl) <: Int
        lvl = lvl:lvl
    end
    pop = Array{Array{Float64,1},1}(undef, length(s))
    for (i, v) in enumerate(s)
        w, v = eigen_decomp(H, v, lvl = maximum(lvl), eig_init = eig_init)
        if ndims(states[i]) == 1
            inst_state = view(v, :, lvl)'
            pop[i] = abs2.(inst_state * states[i])
        elseif ndims(states[i]) == 2
            l = length(lvl)
            temp = Array{Float64,1}(undef, l)
            for j in range(1, length = l)
                inst_state = view(v, :, j)
                temp[j] = real(inst_state' * states[i] * inst_state)
            end
            pop[i] = temp
        end
    end
    pop
end
