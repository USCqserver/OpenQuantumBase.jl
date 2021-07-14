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

Calculate the partial trace of the density matrix `ρ`. Assume `ρ` is in the tensor space of ``ℋ₁⊗ℋ₂⊗…``, `sys_dim` is an array of the corresponding sub-system dimensions. `dim_2_keep` is an array of the indices whose corresponding subsystems are not traced out.

# Examples
```julia-repl
julia> ρ1 = [0.4 0.2; 0.2 0.6]; ρ2 = [0.5 0; 0 0.5];
julia> partial_trace(ρ1⊗ρ2, [2, 2], [1])
2×2 Array{Float64,2}:
 0.4  0.2
 0.2  0.6
```
"""
function partial_trace(ρ::Matrix, sys_dim::AbstractVector{Int}, dim_2_keep::AbstractVector{Int})
    size(ρ, 1) != size(ρ, 2) && throw(ArgumentError("ρ is not a square matrix."))
    prod(sys_dim) != size(ρ, 1) && throw(ArgumentError("System dimensions do not multiply to density matrix dimension."))

    N = length(sys_dim)
    sys_dim = reverse(sys_dim)
    ρ = reshape(ρ, sys_dim..., sys_dim...)
    dim_2_keep = N .+ 1 .- dim_2_keep
    dim_2_trace = [i for i in 1:N if !(i in dim_2_keep)]
    traction_idx = collect(1:2*N)
    traction_idx[dim_2_trace .+ N] .= traction_idx[dim_2_trace]
    tensortrace(ρ, traction_idx)
end

"""
$(SIGNATURES)

Calculate the indices of the degenerate values in list `w`. `w` must be sorted in ascending order.

# Examples
```julia-repl
julia> w = [1,1,1,2,3,4,4,5]
julia> find_degenerate(w)
2-element Array{Array{Int64,1},1}:
 [1, 2, 3]
 [6, 7]
```
"""
function find_degenerate(w; atol::Real=1e-6, rtol::Real=0)
    dw = w[2:end] - w[1:end-1]
    inds = findall((x)->isapprox(x, 0.0, atol=atol, rtol=rtol), dw)
    res = Array{Array{Int, 1}, 1}()
    temp = Int[]
    for i in eachindex(inds)
        push!(temp, inds[i])
        if (i==length(inds)) || !(inds[i]+1 == inds[i+1])
            push!(temp, inds[i]+1)
            push!(res, temp)
            temp = Int[]
        end
    end
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

Check whether the input `ρ` is a pure state: ``tr(ρ²)≈1``. This function will first check if `ρ` is a valid density matrix. `atol` and `rtol` are the corresponding absolute and relative error tolerance used for float point number comparison.
"""
function check_pure_state(ρ; atol::Real=0, rtol::Real=atol>0 ? 0 : √eps())
    check_density_matrix(ρ) && isapprox(purity(ρ), 1, atol=atol, rtol=rtol)
end