"""
$(SIGNATURES)

Evaluate the time dependent Hamiltonian at time s with the unit of `GHz`. 

Fallback to `H.(s)/2/π` for generic `AbstractHamiltonian` type.
"""
evaluate(H::AbstractHamiltonian, s::Real) = H.(s) / 2 / π

function (H::AbstractHamiltonian, p, s::Real) 
    H.(s) / 2 / π
end

"""
isconstant(H)

Check whether a Hamiltonian is constant.
"""
isconstant(::AbstractHamiltonian) = false

"""
$(SIGNATURES)

Update the internal cache `cache` according to the value of the Hamiltonian `H` at given dimensionless time `s`: ``cache = -iH(p, s)``. The third argument, `p` is reserved for passing additional info to the `AbstractHamiltonian` object. Currently, it is only used by `AdiabaticFrameHamiltonian` to pass the total evolution time `tf`. To keep the interface consistent across all `AbstractHamiltonian` types, the `update_cache!` method for all subtypes of `AbstractHamiltonian` should keep the argument `p`.

Fallback to `cache .= -1.0im * H(p, s)` for generic `AbstractHamiltonian` type.
"""
update_cache!(cache, H::AbstractHamiltonian, p, s::Real) = cache .= -1.0im * H(p, s)

function update_vectorized_cache!(cache, H::AbstractHamiltonian, p, s::Real)
    hmat = H(s)
    iden = one(hmat)
    cache .= 1.0im * (transpose(hmat) ⊗ iden - iden ⊗ hmat)
end

function (h::AbstractHamiltonian)(du, u::AbstractMatrix, p, s::Real)
    H = h(s)
    Hρ = -1.0im * H * u
    du .= Hρ - transpose(Hρ)
end

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

    Default eigenvalue decomposition method for an abstract Hamiltonian `H` at time `s`. Requires the Hamiltonian to be callable and have a `u_cache` field. Keyword argument `lvl` specifies the number of levels to keep in the output. 
    The function returns (w, v), where `w` is a vector of eigenvalues and `v` is a matrix of the eigenvectors in the columns. 
    (The `k`th eigenvector can be obtained from the slice `v[:, k]`.)
"""
haml_eigs_default(H::AbstractHamiltonian, t, lvl::Integer) = eigen!(Hermitian(H(t)), 1:lvl)
haml_eigs_default(H::AbstractHamiltonian, t, ::Nothing) = eigen(Hermitian(H(t)))
haml_eigs(H::AbstractHamiltonian, t, lvl) = haml_eigs_default(H, t, lvl)
# Default eigendecomposition routine for AbstractSparseHamiltonian Hamiltonian
# It converts all sparse matrices into dense ones and use `LAPACK` routine for eigendecomposition
haml_eigs_default(H::AbstractSparseHamiltonian, t, ::Nothing) = eigen!(Hermitian(Array(H(t))))
haml_eigs_default(H::AbstractSparseHamiltonian, t, lvl::Integer) = eigen!(Hermitian(Array(H(t))), 1:lvl)

#function eigen!(M::Hermitian{T, S}, lvl::UnitRange) where T<:Number where S<:Union{SMatrix, MMatrix}
#    w, v = eigen(Hermitian(M))
#    w[lvl], v[:, lvl]
#end

"""
$(SIGNATURES)

Calculate the eigen value decomposition of the Hamiltonian `H` at time `s`. Keyword argument `lvl` specifies the number of levels to keep in the output. `w` is a vector of eigenvalues and `v` is a matrix of the eigenvectors in the columns. (The `k`th eigenvector can be obtained from the slice `v[:, k]`.) `w` will be in unit of `GHz`.
"""
function eigen_decomp(H::AbstractHamiltonian, s; lvl::Int=2)
    w, v = haml_eigs(H, s, lvl)
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
        val, vec = haml_eigs(H, s_val, lvl)
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

function is_complex(f_list, m_list)
    any(m_list) do m
        eltype(m) <: Complex
    end || any(f_list) do f
        typeof(f(0)) <: Complex
    end
end

function ConstantHamiltonian(mat::Matrix; unit=:h, static=true)
    # use static array for size smaller than 100
    # can be turned off by setting `static` to false
    if static && size(mat, 1) <= 10
        mat = SMatrix{size(mat, 1), size(mat, 2)}(mat)
        return ConstantStaticDenseHamiltonian(mat, unit=unit)
    end
    ConstantDenseHamiltonian(mat, unit=unit)
end

ConstantHamiltonian(mat::Union{SMatrix,MMatrix}; unit=:h, static=true)=ConstantStaticDenseHamiltonian(mat, unit=unit)

function ConstantHamiltonian(mat::SparseMatrixCSC; unit=:h, static=true)
    mat = unit_scale(unit) * mat
    ConstantSparseHamiltonian(mat, size(mat))
end

function Hamiltonian(f, mats; unit=:h, dimensionless_time=true, static=true)
    @warn "Elements in `mats` have different types. Will attempt to promote them to a common type."
    mats = promote(mats)
    Hamiltonian(f, mats, unit=unit, dimensionless_time=dimensionless_time, static= static)
end

function Hamiltonian(f, mats::AbstractVector{T}; unit=:h, dimensionless_time=true, static=true) where {T<:Matrix}
    hsize = size(mats[1])
    # use static array for size smaller than 100
    # can be turned off by setting `static` to false
    if static && hsize[1] <= 10
        mats = [SMatrix{hsize[1],hsize[2]}(unit_scale(unit) * m) for m in mats]
        return StaticDenseHamiltonian(f, mats, unit=unit, dimensionless_time=dimensionless_time)
    end
    DenseHamiltonian(f, mats, unit=unit, dimensionless_time=dimensionless_time)
end

Hamiltonian(f, mats::AbstractVector{T}; unit=:h, dimensionless_time=true, static=true) where {T<:Union{SMatrix, MMatrix}} = StaticDenseHamiltonian(f, mats, unit=unit, dimensionless_time=dimensionless_time)

Hamiltonian(f, mats::AbstractVector{T}; unit=:h, dimensionless_time=true, static=true) where {T<:SparseMatrixCSC} = SparseHamiltonian(f, mats, unit=unit, dimensionless_time=dimensionless_time)