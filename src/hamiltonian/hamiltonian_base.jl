"""
$(SIGNATURES)

Evaluate the time dependent Hamiltonian at time s with the unit of `GHz`
"""
evaluate(H::AbstractHamiltonian, s::Real) = H.(s) / 2 / π

"""
$(SIGNATURES)

Update the internal cache `cache` according to the value of the Hamiltonian `H` at given dimensionless time `s`: ``cache = -iH(p, s)``. The third argument, `p` is reserved for passing additional info to the `AbstractHamiltonian` object. Currently, it is only used by `AdiabaticFrameHamiltonian` to pass the total evolution time `tf`. To keep the interface consistent across all `AbstractHamiltonian` types, the `update_cache!` method for all subtypes of `AbstractHamiltonian` should keep the argument `p`.

Fallback to `cache .= -1.0im * H(p, s)` for generic `AbstractHamiltonian` type.
"""
update_cache!(cache, H::AbstractHamiltonian, p, s::Real) = cache .= -1.0im * H(p, s)

"""
$(SIGNATURES)
The `AbstractHamiltonian` type can be called with two arguments. The first argument is reserved to pass additional info to the `AbstractHamiltonian` object. Currently, it is only used by `AdiabaticFrameHamiltonian` to pass the total evolution time `tf`.

Fallback to `H(s)` for generic `AbstractHamiltonian` type.
"""
(H::AbstractHamiltonian)(::Any, s::Real) = H(s)

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
$(SIGNATURES)

    Default eigenvalue decomposition method for an abstract Hamiltonian `H` at time `s`. 
    Requires the Hamiltonian to be callable and have a u_cache field
    Keyword argument `lvl` specifies the number of levels to keep in the output. 
    `w` is a vector of eigenvalues and `v` is a matrix of the eigenvectors in the columns. 
    (The `k`th eigenvector can be obtained from the slice `v[:, k]`.)
"""
function haml_eigs_default(H::AbstractHamiltonian, t; lvl::Union{Int,Nothing}=nothing)
    #lvl = size(H.u_cache, 1)
    if isnothing(lvl)
        w,v = eigen(Hermitian(H(t)))
        return real(w), v
    elseif lvl <= 10
        w,v = eigen(Hermitian(H(t)))
    else
        w,v = eigen!(Hermitian(H(t)), 1:lvl)
    end
    
    return real(w)[1:lvl], v[:, 1:lvl]
end

function haml_eigs(H::AbstractHamiltonian, t; lvl::Union{Int,Nothing}=nothing)
    return haml_eigs_default(H, t, lvl=lvl)
end

"""
$(SIGNATURES)

Calculate the eigen value decomposition of the Hamiltonian `H` at time `s`. Keyword argument `lvl` specifies the number of levels to keep in the output. `w` is a vector of eigenvalues and `v` is a matrix of the eigenvectors in the columns. (The `k`th eigenvector can be obtained from the slice `v[:, k]`.) `w` will be in unit of `GHz`.
"""
function eigen_decomp(H::AbstractHamiltonian, s; lvl::Int=2)
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
    lvl::Int=2
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
"""
function inst_population(s, states, H::AbstractHamiltonian; lvl=1:1)
    if typeof(lvl) <: Int
        lvl = lvl:lvl
    end
    pop = Array{Array{Float64,1},1}(undef, length(s))
    for (i, v) in enumerate(s)
        w, v = eigen_decomp(H, v, lvl=maximum(lvl))
        if ndims(states[i]) == 1
            inst_state = view(v, :, lvl)'
            pop[i] = abs2.(inst_state * states[i])
        elseif ndims(states[i]) == 2
            l = length(lvl)
            temp = Array{Float64,1}(undef, l)
            for j in range(1, length=l)
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

EIGEN_DEFAULT(::SparseMatrixCSC) =
    (H, t, lvl) -> eigen!(Hermitian(Array(H(t))), 1:lvl)

function is_complex(f_list, m_list)
    any(m_list) do m
        eltype(m) <: Complex
    end || any(f_list) do f
        typeof(f(0)) <: Complex
    end
end