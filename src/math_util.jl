import LinearAlgebra: kron, eigmin

const σx = [0.0 + 0.0im 1; 1 0]
const σy = [0.0 + 0.0im -1.0im; 1.0im 0]
const σz = [1.0 + 0.0im 0; 0 -1]
const σi = [1.0 + 0.0im 0; 0 1]
const σ₋ = [0.0im 0;1 0]
const σ₊ = [0.0im 1;0 0]
const σ = [σx, σy, σz, σi]
xvec = [[1.0 + 0.0im, 1.0] / sqrt(2), [1.0 + 0.0im, -1.0] / sqrt(2)]
yvec = [[1.0, 1.0im] / sqrt(2), [1.0, -1.0im] / sqrt(2)]
zvec = [[1.0 + 0.0im, 0], [0, 1.0 + 0.0im]]

const spσz = sparse(σz)
const spσx = sparse(σx)
const spσy = sparse(σy)
const spσi = sparse(I, 2, 2)

"""
    PauliVec

A constant that holds the eigenvectors of the Pauli matrices. Indices 1, 2 and 3 corresponds to the eigenvectors of ``σ_x``, ``σ_y`` and ``σ_z``.

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
$(SIGNATURES)

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
function matrix_decompose(mat::AbstractMatrix, basis::Vector{<:AbstractMatrix})
    dim = size(basis[1])[1]
    [tr(mat * b) / dim for b in basis]
end

"""
    check_positivity(m)

Check if matrix `m` is positive. Return `true` is the minimum eigenvalue of `m` is greater than or equal to 0.
"""
function check_positivity(m::AbstractMatrix)
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
    β = temperature_2_β(T)
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
$(SIGNATURES)

Test if `𝐔` is a unitary matrix. The function checks how close both `` 𝐔𝐔^† `` and `` 𝐔^†𝐔 `` are to `` I ``, with relative and absolute error given by `rtol`, `atol`.

# Examples
```julia-repl
julia> check_unitary(exp(-1.0im*5*0.5*σx))
true
```
"""
function check_unitary(𝐔::AbstractMatrix; rtol = 1e-6, atol = 1e-8)
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

"""
$(SIGNATURES)

Generate a log-uniformly distributed array with `num` elements between `a` and `b`. The base of log is `base` with default value 10.
"""
function log_uniform(a, b, num; base = 10)
    loga = log(base, a)
    logb = log(base, b)
    10 .^ range(loga, logb, length = num)
end

"""
$(SIGNATURES)

Calculate the partial trace of the density matrix `ρ`, assuming all the subsystems are qubits. `qubit_2_keep` denotes the indices whose corresponding qubits are not traced out.

# Examples
```julia-repl
julia> ρ1 = [0.4 0.2; 0.2 0.6]; ρ2 = [0.5 0; 0 0.5];
julia> partial_trace(ρ1⊗ρ2, [1])
2×2 Array{Float64,2}:
 0.4  0.2
 0.2  0.6
```
"""
function partial_trace(ρ::Matrix, qubit_2_keep::AbstractVector{Int})
    num = Int(log2(size(ρ, 1)))
    partial_trace(ρ, [2 for i in 1:num], qubit_2_keep)
end

"""
$(SIGNATURES)

Calculate the partial trace of the density matrix `ρ`. Assume `ρ` is in the tensor space of ``ℋ₁⊗ℋ₂⊗…``,
`sys_dim` is an array of the corresponding sub-system dimensions. `dim_2_keep` is an array of the indices
whose corresponding subsystems are not traced out.

# Examples
```julia-repl
julia> ρ1 = [0.4 0.2; 0.2 0.6]; ρ2 = [0.5 0; 0 0.5];
julia> partial_trace(ρ1⊗ρ2, [2, 2], [1])
2×2 Array{Float64,2}:
 0.4  0.2
 0.2  0.6
```
"""
partial_trace(ρ::AbstractMatrix, sys_dim::AbstractVector{<:Integer}, dim_2_keep::Integer) = partial_trace(ρ,sys_dim,[dim_2_keep])

function partial_trace(ρ::AbstractMatrix, sys_dim::AbstractVector{<:Integer}, dim_2_keep::AbstractVector{<:Integer})
    size(ρ, 1) != size(ρ, 2) && throw(ArgumentError("ρ is not a square matrix."))
    prod(sys_dim) != size(ρ, 1) && throw(ArgumentError("System dimensions do not multiply to density matrix dimension."))

	N = length(sys_dim)
	iden = CartesianIndex(ones(Int, 2*N)...)
	sys_dim = reverse(sys_dim)
	ρ = reshape(ρ, sys_dim..., sys_dim...)
	dim_2_keep = N .+ 1 .- dim_2_keep
	dim_2_trace = [i for i in 1:N if !(i in dim_2_keep)]
	re_dim = copy(sys_dim)
	re_dim[dim_2_trace] .= 1
	tr_dim = copy(sys_dim)
	tr_dim[dim_2_keep] .= 1
	res = zeros(typeof(first(ρ)), re_dim..., re_dim...)
	for I in CartesianIndices(size(res))
		for k in CartesianIndices((tr_dim...,))
			delta = CartesianIndex(k, k)
			res[I] += ρ[I + delta - iden]
		end
	end
	reshape(res, sys_dim[dim_2_keep]..., sys_dim[dim_2_keep]...)
end

"""
$(SIGNATURES)

Calculate the indices of the degenerate values in list `w`. `w` must be sorted in ascending order.

If the `digits` keyword argument is provided, it rounds the energies to the specified number of digits after the decimal place (or before if negative), in base 10. The default value is 8.

# Examples
```julia-repl
julia> w = [1,1,1,2,3,4,4,5]
julia> find_degenerate(w)
2-element Array{Array{Int64,1},1}:
 [1, 2, 3]
 [6, 7]
```
"""
function find_degenerate(w; digits::Integer=8)
    w = round.(w, digits = digits)
    res = []
    tmp = []
    flag = true
    for i in 2:length(w)
        if w[i] == w[i-1] && flag
            push!(tmp, i-1, i)
            flag = false
        elseif w[i] == w[i-1]
            push!(tmp, i)
        elseif !flag
            push!(res, tmp)
            tmp = []
            flag = true
        end
    end
    isempty(tmp) ? nothing : push!(res, tmp)
    res
end

"""
$(SIGNATURES)

Calculate the fidelity between two density matrices `ρ` and `σ`: ``Tr[\\sqrt{\\sqrt{ρ}σ\\sqrt{ρ}}]²``.

# Examples
```julia-repl
julia> ρ = PauliVec[1][1]*PauliVec[1][1]'
julia> σ = PauliVec[3][1]*PauliVec[3][1]'
julia> fidelity(ρ, σ)
0.49999999999999944
```
"""
function fidelity(ρ, σ)
    if !check_density_matrix(ρ) || !check_density_matrix(σ)
        throw(ArgumentError("Inputs are not valid density matrices."))
    end
    temp = sqrt(ρ)
    real(tr(sqrt(temp * σ * temp))^2)
end

"""
$(SIGNATURES)

Check whether matrix `ρ` is a valid density matrix. `atol` and `rtol` is the absolute and relative error tolerance to use when checking `ρ` ≈ 1.

# Examples
```julia-repl
julia> check_density_matrix(σx)
false
julia> check_density_matrix(σi/2)
true
```
"""
function check_density_matrix(ρ; atol::Real=0, rtol::Real=atol>0 ? 0 : √eps())
    ishermitian(ρ) && (eigmin(ρ) >= 0) && isapprox(tr(ρ), 1, atol=atol, rtol=rtol)
end

"""
$(SIGNATURES)

Calculate the purity of density matrix `ρ`: ``tr(ρ²)``. This function does not check whether `ρ` is a valid density matrix.
"""
function purity(ρ)
    tr(ρ*ρ)
end

"""
$(SIGNATURES)

Check whether the input `ρ` is a pure state: ``tr(ρ²)≈1``. This function will first check if `ρ` is a valid density matrix. `atol` and `rtol` are the corresponding absolute and relative error tolerance used for float point number comparison. The default values are `atol = 0` and `rtol = atol>0 ? 0 : √eps`.
"""
function check_pure_state(ρ; atol::Real=0, rtol::Real=atol>0 ? 0 : √eps())
    check_density_matrix(ρ) && isapprox(purity(ρ), 1, atol=atol, rtol=rtol)
end

"""
$(SIGNATURES)

Generate a Haar random unitary matrix of dimension `(dim, dim)` using the QR decomposition method.

# Arguments
- `dim::Int`: the dimension of the matrix.

# Returns
- A Haar random unitary matrix of size `(dim, dim)`.

# Details
A Haar random unitary matrix is a matrix whose entries are complex numbers of modulus 1, chosen randomly with respect to the Haar measure on the unitary group U(dim). This implementation uses the QR decomposition method, where a complex Gaussian matrix of size `(dim, dim)` is generated, and then it is QR decomposed into the product of an orthogonal matrix and an upper triangular matrix with positive diagonal entries. The diagonal entries are then normalized by their absolute values to obtain the unitary matrix.

# Examples
```julia
julia> haar_unitary(3)
3×3 Matrix{ComplexF64}:
  0.407398+0.544041im   0.799456-0.213793im  -0.439273+0.396871im
 -0.690274-0.288499im   0.347444+0.446328im   0.426262+0.505472im
  0.437266-0.497024im  -0.067918-0.582238im   0.695026+0.186022im
```
"""
function haar_unitary(dim)
   M = randn(dim,dim)+im*randn(dim, dim)
   q,r = qr(M)
   L = diag(r)
   L=L./abs.(L)
   q*diagm(0=>L)
end
