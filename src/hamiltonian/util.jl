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
$(SIGNATURES)

Calculate the eigen value decomposition of the Hamiltonian `H` at time `s`. Keyword argument `lvl` specifies the number of levels to keep in the output. `w` is a vector of eigenvalues and `v` is a matrix of the eigenvectors in the columns. (The `k`th eigenvector can be obtained from the slice `v[:, k]`.) `w` will be in unit of `GHz`.

`eig_init` is the initializer for eigen factorization routine. It returns a function of signature: `(H, s, lvl) -> (w, v)`. The default initializer `EIGEN_DEFAULT` will use `LAPACK` routine for both dense and sparse matrices.
"""
function eigen_decomp(H::AbstractHamiltonian, s; lvl::Int = 2)
    w, v = H.EIGS(H, s, lvl)
    real(w)[1:lvl] / 2 / π, v[:, 1:lvl]
end


"""
$(SIGNATURES)

Calculate the eigen value decomposition of the Hamiltonian `H` at an array of time points `s`. The output keeps the lowest `lvl` eigenstates and their corresponding eigenvalues. Output `(vals, vecs)` have the dimensions of `(lvl, length(s))` and `(size(H, 1), lvl, length(s))` respectively.
"""
function eigen_decomp(
    H::AbstractHamiltonian,
    s::AbstractArray{Float64,1};
    lvl::Int = 2
)
    s_dim = length(s)
    res_val = Array{eltype(H),2}(undef, (lvl, s_dim))
    res_vec = Array{eltype(H),3}(undef, (size(H, 1), lvl, s_dim))
    for (i, s_val) in enumerate(s)
        val, vec = H.EIGS(H, s_val, lvl)
        res_val[:, i] = val[1:lvl]
        res_vec[:, :, i] = vec[:, 1:lvl]
    end
    res_val, res_vec
end


"""
$(SIGNATURES)

For a time series quantum states given by `states`, whose time points are given by `s`, calculate the population of instantaneous eigenstates of `H`. The levels of the instantaneous eigenstates are specified by `lvl`, which can be any slice index.

`eig_init` is the initializer for eigen factorization routine. It returns a function of signature: `(H, s, lvl) -> (w, v)`. The default initializer `EIGEN_DEFAULT` will use `LAPACK` routine for both dense and sparse matrices.
"""
function inst_population(s, states, H::AbstractHamiltonian; lvl = 1:1)
    if typeof(lvl) <: Int
        lvl = lvl:lvl
    end
    pop = Array{Array{Float64,1},1}(undef, length(s))
    for (i, v) in enumerate(s)
        w, v = eigen_decomp(H, v, lvl = maximum(lvl))
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

"""
    function EIGEN_DEFAULT(u_cache)

The default initializer for eigen factorization method. It returns a function of signature: `(H, s, lvl) -> (w, v)`. `u_cache` is the cache for Hamiltonian object, `s` is the time argument for the Hamiltonian and `lvl` is the energy levels to keep. This default initializer will use `LAPACK` routine for both dense and sparse matrices.
"""
function EIGEN_DEFAULT(u_cache)
    lvl = size(u_cache, 1)
    if lvl <= 10
        EIGS = (H, t, lvl) -> eigen(Hermitian(H(t)))
    else
        EIGS = (H, t, lvl) -> eigen!(Hermitian(H(t)), 1:lvl)
    end
    EIGS
end

EIGEN_DEFAULT(u_cache::SparseMatrixCSC) =
    (H, t, lvl) -> eigen!(Hermitian(Array(H(t))), 1:lvl)
