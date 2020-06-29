import LinearAlgebra: kron, eigmin

const σx = [0.0 + 0.0im 1; 1 0]
const σy = [0.0 + 0.0im -1.0im; 1.0im 0]
const σz = [1.0 + 0.0im 0; 0 -1]
const σi = [1.0 + 0.0im 0; 0 1]
const σ = [σx, σy, σz, σi]
xvec = [[1.0 + 0.0im, 1.0] / sqrt(2), [1.0 + 0.0im, -1.0] / sqrt(2)]
yvec = [[1.0im, -1.0] / sqrt(2), [1.0im, 1.0] / sqrt(2)]
zvec = [[1.0 + 0.0im, 0], [0, 1.0 + 0.0im]]

const spσz = sparse(σz)
const spσx = sparse(σx)
const spσy = sparse(σy)
const spσi = sparse(I, 2, 2)

"""
    PauliVec

Constants for the eigenvectors of single qubit Pauli matrices. Indices 1, 2 and 3 corresponds to the eigenvectors of ``σ_x``, ``σ_y`` and ``σ_z``.

# Examples
```julia-repl
julia> σx*PauliVec[1][1] == PauliVec[1][1]
true
```
"""
const PauliVec = [xvec, yvec, zvec]

"""
    ⊗(A, B)

Calculate the tensor product of `A` and `B`.

# Examples
```julia-repl
julia> σx⊗σz
4×4 Array{Complex{Float64},2}:
 0.0+0.0im   0.0+0.0im  1.0+0.0im   0.0+0.0im
 0.0+0.0im  -0.0+0.0im  0.0+0.0im  -1.0+0.0im
 1.0+0.0im   0.0+0.0im  0.0+0.0im   0.0+0.0im
 0.0+0.0im  -1.0+0.0im  0.0+0.0im  -0.0+0.0im
```
"""
⊗ = kron

"""
    matrix_decompose(mat::Matrix{T}, basis::Array{Matrix{T},1})

Decompse matrix `mat` onto matrix basis `basis`

# Examples
```julia-repl
julia> matrix_decompose(1.0*σx+2.0*σy+3.0*σz, [σx,σy,σz])
3-element Array{Complex{Float64},1}:
 1.0 + 0.0im
 2.0 + 0.0im
 3.0 + 0.0im
```
"""
function matrix_decompose(
    mat::Matrix{T},
    basis::Array{Matrix{T},1},
) where {T<:Number}
    dim = size(basis[1])[1]
    [tr(mat * b) / dim for b in basis]
end

"""
    check_positivity(m)

Check if matrix `m` is positive. Return `true` is the minimum eigenvalue of `m` is greater than or equal to 0.
"""
function check_positivity(m::Matrix{T}) where {T<:Number}
    if !ishermitian(m)
        @warn "Input fails the numerical test for Hermitian matrix. Use the upper triangle to construct a new Hermitian matrix."
        d = Hermitian(m)
    else
        d = m
    end
    eigmin(d) >= 0
end


"""
    gibbs_state(h, β)

Calculate the Gibbs state of the matrix `h` at temperature `T` (mK).

# Examples
```julia-repl
julia> gibbs_state(σz, 10)
2×2 Array{Complex{Float64},2}:
 0.178338+0.0im       0.0+0.0im
      0.0+0.0im  0.821662+0.0im
```
"""
function gibbs_state(h, T)
    β = temperature_2_beta(T)
    Z = 0.0
    w, v = eigen(Hermitian(h))
    res = zeros(eltype(v), size(h))
    for (i, E) in enumerate(w)
        t = exp(-β * E)
        vi = @view v[:, i]
        res += t * vi * vi'
        Z += t
    end
    res / Z
end

"""
    low_level_matrix(M, lvl)

Calculate the matrix `M` projected to lower energy subspace containing `lvl` energy lvl.

# Examples
```julia-repl
julia> low_level_matrix(σx⊗σx, 2)
4×4 Array{Complex{Float64},2}:
 -0.5+0.0im   0.0+0.0im   0.0+0.0im   0.5+0.0im
  0.0+0.0im  -0.5+0.0im   0.5+0.0im   0.0+0.0im
  0.0+0.0im   0.5+0.0im  -0.5+0.0im   0.0+0.0im
  0.5+0.0im   0.0+0.0im   0.0+0.0im  -0.5+0.0im
```
"""
function low_level_matrix(M, lvl)
    if lvl > size(M, 1)
        @warn "Subspace dimension bigger than total dimension."
        return M
    else
        w, v = eigen(Hermitian(M))
        res = zeros(eltype(v), size(M))
        for i in range(1, stop = lvl)
            vi = @view v[:, i]
            res += w[i] * vi * vi'
        end
        return res
    end
end


"""
    check_unitary(𝐔; rtol=1e-6, atol=1e-8)

Test if `𝐔` is a unitary matrix. The function checks how close both `` 𝐔𝐔^† `` and `` 𝐔^†𝐔 `` are to `` I ``, with relative and absolute error given by `rtol`, `atol`.

# Examples
```julia-repl
julia> check_unitary(exp(-1.0im*5*0.5*σx))
true
```
"""
function check_unitary(𝐔::Matrix{T}; rtol = 1e-6, atol = 1e-8) where {T<:Number}
    a1 = isapprox(
        𝐔 * 𝐔',
        Matrix{eltype(𝐔)}(I, size(𝐔)),
        rtol = rtol,
        atol = atol,
    )
    a2 = isapprox(
        𝐔' * 𝐔,
        Matrix{eltype(𝐔)}(I, size(𝐔)),
        rtol = rtol,
        atol = atol,
    )
    a1 && a2
end
