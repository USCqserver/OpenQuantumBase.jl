import LinearAlgebra: normalize

"""
    q_translate(h::String; sp = false)

Convert a string `h` representing multi-qubits Pauli matrices summation into its numerical form. Generate sparse matrix when `sp` is set to true.

# Examples
```julia-repl
julia> q_translate("X+2.0Z")
2×2 Array{Complex{Float64},2}:
 2.0+0.0im   1.0+0.0im
 1.0+0.0im  -2.0+0.0im
```
"""
function q_translate(h::String; sp = false)
    # define operator map to replace [XYZI] with corresponding Pauli matrices
    function operator_map(x)
        if sp == false
            σ_tag = "σ"
        else
            σ_tag = "spσ"
        end
        res = ""
        for i in range(1, length = length(x) - 1)
            res = res * σ_tag * lowercase(x[i]) * "⊗"
        end
        res = res * σ_tag * lowercase(x[end])
    end

    h_str = replace(h, r"[XYZI]+" => operator_map)
    eval(Meta.parse(h_str))
end

"""
    single_clause(ops::Vector{String}, q_ind, weight, num_qubit; sp=false)

Construct a single clause of the multi-qubits Hamiltonian. `ops` is a list of Pauli operator names which appear in this clause. `q_ind` is the list of indices corresponding to the Pauli matrices in `ops`. `weight` is the constant factor of this clause. `num_qubit` is the total number of qubits. A sparse matrix can be construct by setting `sp` to `true`. The following example construct a clause of `` Z_1 I Z_3/2 ``.

# Examples
```julia-repl
julia> single_clause(["z", "z"], [1, 3], 0.5, 3)
8×8 Array{Complex{Float64},2}:
 0.5+0.0im   0.0+0.0im  0.0+0.0im   0.0+0.0im   0.0+0.0im   0.0+0.0im   0.0+0.0im   0.0+0.0im
 0.0+0.0im  -0.5+0.0im  0.0+0.0im  -0.0+0.0im   0.0+0.0im  -0.0+0.0im   0.0+0.0im  -0.0+0.0im
 0.0+0.0im   0.0+0.0im  0.5+0.0im   0.0+0.0im   0.0+0.0im   0.0+0.0im   0.0+0.0im   0.0+0.0im
 0.0+0.0im  -0.0+0.0im  0.0+0.0im  -0.5+0.0im   0.0+0.0im  -0.0+0.0im   0.0+0.0im  -0.0+0.0im
 0.0+0.0im   0.0+0.0im  0.0+0.0im   0.0+0.0im  -0.5+0.0im  -0.0+0.0im  -0.0+0.0im  -0.0+0.0im
 0.0+0.0im  -0.0+0.0im  0.0+0.0im  -0.0+0.0im  -0.0+0.0im   0.5-0.0im  -0.0+0.0im   0.0-0.0im
 0.0+0.0im   0.0+0.0im  0.0+0.0im   0.0+0.0im  -0.0+0.0im  -0.0+0.0im  -0.5+0.0im  -0.0+0.0im
 0.0+0.0im  -0.0+0.0im  0.0+0.0im  -0.0+0.0im  -0.0+0.0im   0.0-0.0im  -0.0+0.0im   0.5-0.0im
```
"""
function single_clause(ops::Vector{String}, q_ind, weight, num_qubit; sp = false)
    if sp == false
        σ_tag = "σ"
        i_tag = σi
    else
        σ_tag = "spσ"
        i_tag = spσi
    end
    res = weight
    for i = 1:num_qubit
        idx = findfirst((x) -> x == i, q_ind)
        if !isnothing(idx)
            op2 = eval(Meta.parse(σ_tag * lowercase(ops[idx])))
            res = res ⊗ op2
        else
            res = res ⊗ i_tag
        end
    end
    res
end

"""
    single_clause(ops::Vector{T}, q_ind, weight, num_qubit; sp=false) where T<:AbstractMatrix

Construct a single clause of the multi-qubits Hamiltonian. `ops` is a list of single-qubit operators that appear in this clause. The sparse identity matrix is used once `sp` is set to `true`. The following example construct a clause of `` Z_1 I Z_3/2 ``.

# Examples
```julia-repl
julia> single_clause([σz, σz], [1, 3], 0.5, 3)
8×8 Array{Complex{Float64},2}:
 0.5+0.0im   0.0+0.0im  0.0+0.0im   0.0+0.0im   0.0+0.0im   0.0+0.0im   0.0+0.0im   0.0+0.0im
 0.0+0.0im  -0.5+0.0im  0.0+0.0im  -0.0+0.0im   0.0+0.0im  -0.0+0.0im   0.0+0.0im  -0.0+0.0im
 0.0+0.0im   0.0+0.0im  0.5+0.0im   0.0+0.0im   0.0+0.0im   0.0+0.0im   0.0+0.0im   0.0+0.0im
 0.0+0.0im  -0.0+0.0im  0.0+0.0im  -0.5+0.0im   0.0+0.0im  -0.0+0.0im   0.0+0.0im  -0.0+0.0im
 0.0+0.0im   0.0+0.0im  0.0+0.0im   0.0+0.0im  -0.5+0.0im  -0.0+0.0im  -0.0+0.0im  -0.0+0.0im
 0.0+0.0im  -0.0+0.0im  0.0+0.0im  -0.0+0.0im  -0.0+0.0im   0.5-0.0im  -0.0+0.0im   0.0-0.0im
 0.0+0.0im   0.0+0.0im  0.0+0.0im   0.0+0.0im  -0.0+0.0im  -0.0+0.0im  -0.5+0.0im  -0.0+0.0im
 0.0+0.0im  -0.0+0.0im  0.0+0.0im  -0.0+0.0im  -0.0+0.0im   0.0-0.0im  -0.0+0.0im   0.5-0.0im
```
"""
function single_clause(ops::Vector{T}, q_ind, weight, num_qubit; sp = false) where T<:AbstractMatrix
    i_tag = sp == false ? σi : spσi
    res = weight
    for i = 1:num_qubit
        idx = findfirst((x) -> x == i, q_ind)
        if !isnothing(idx)
            res = res ⊗ ops[idx]
        else
            res = res ⊗ i_tag
        end
    end
    res
end

"""
    collective_operator(op, num_qubit; sp=false)

Construct the collective operator for a system of `num_qubit` qubits. `op` is the name of the collective Pauli matrix. For example, the following code construct an `` IZ + ZI `` matrix. Generate sparse matrix when `sp` is set to true.

# Examples
```julia-repl
julia> collective_operator("z", 2)
4×4 Array{Complex{Float64},2}:
 2.0+0.0im  0.0+0.0im  0.0+0.0im   0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im   0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im   0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  -2.0+0.0im
```
"""
function collective_operator(op, num_qubit; sp = false)
    op_name = uppercase(op)
    res = ""
    for idx = 1:num_qubit
        res = res * "I"^(idx - 1) * op_name * "I"^(num_qubit - idx) * "+"
    end
    q_translate(res[1:end-1]; sp = sp)
end

"""
    standard_driver(num_qubit; sp=false)

Construct the standard driver Hamiltonian for a system of `num_qubit` qubits. For example, a two qubits standard driver matrix is `` IX + XI ``. Generate sparse matrix when `sp` is set to true.
"""
function standard_driver(num_qubit; sp = false)
    res = ""
    for idx = 1:num_qubit
        res = res * "I"^(idx - 1) * "X" * "I"^(num_qubit - idx) * "+"
    end
    q_translate(res[1:end-1], sp = sp)
end


"""
    hamming_weight_operator(num_qubit::Int64, op::String; sp=false)

Construct the Hamming weight operator for system of size `num_qubit`. The type of the Hamming weight operator is specified by op: "x", "y" or "z". Generate sparse matrix when `sp` is set to true.

# Examples
```julia-repl
julia> hamming_weight_operator(2,"z")
4×4 Array{Complex{Float64},2}:
 0.0+0.0im  0.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  1.0+0.0im  0.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  1.0+0.0im  0.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im  2.0+0.0im
```
"""
function hamming_weight_operator(num_qubit::Int64, op::String; sp = false)
    0.5 *
    (num_qubit * I - collective_operator(op, num_qubit = num_qubit, sp = sp))
end

"""
    local_field_term(h, idx, num_qubit; sp=false)

Construct local Hamiltonian of the form ``∑hᵢσᵢᶻ``. `idx` is the index of all local field terms and `h` is a list of the corresponding weights. `num_qubit` is the total number of qubits. Generate sparse matrix when `sp` is set to true.

# Examples
```julia-repl
julia> local_field_term([1.0, 0.5], [1, 2], 2) == σz⊗σi+0.5σi⊗σz
true
```
"""
function local_field_term(h, idx, num_qubit; sp = false)
    res = single_clause(["z"], [idx[1]], h[1], num_qubit, sp = sp)
    for i = 2:length(idx)
        res += single_clause(["z"], [idx[i]], h[i], num_qubit, sp = sp)
    end
    res
end


"""
    two_local_term(j, idx, num_qubit; sp=false)

Construct local Hamiltonian of the form ``∑Jᵢⱼσᵢᶻσⱼᶻ``. `idx` is the index of all two local terms and `j` is a list of the corresponding weights. `num_qubit` is the total number of qubits. Generate sparse matrix when `sp` is set to true.

# Examples
```julia-repl
julia> two_local_term([1.0, 0.5], [[1,2], [1,3]], 3) == σz⊗σz⊗σi + 0.5σz⊗σi⊗σz
true
```
"""
function two_local_term(j, idx, num_qubit; sp = false)
    res = single_clause(["z", "z"], idx[1], j[1], num_qubit, sp = sp)
    for i = 2:length(idx)
        res += single_clause(["z", "z"], idx[i], j[i], num_qubit, sp = sp)
    end
    res
end


"""
    q_translate_state(h::String; normal=false)

Convert a string representation of quantum state to a vector. The keyword argument `normal` indicates whether to normalize the output vector. (Currently only '0' and '1' are supported)

# Examples
Single term:
```julia-repl
julia> q_translate_state("001")
8-element Array{Complex{Float64},1}:
 0.0 + 0.0im
 1.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
 0.0 + 0.0im
```
Multiple terms:
```julia-repl
julia> q_translate_state("(101)+(001)", normal=true)
8-element Array{Complex{Float64},1}:
                0.0 + 0.0im
 0.7071067811865475 + 0.0im
                0.0 + 0.0im
                0.0 + 0.0im
                0.0 + 0.0im
 0.7071067811865475 + 0.0im
                0.0 + 0.0im
                0.0 + 0.0im
```
"""
function q_translate_state(h::String; normal = false)
    # TODO: add "+", "-" into the symbol list
    if occursin("(", h) || occursin(")", h)
        h_str = replace(
            h,
            r"\(([01]+)\)" =>
                (x) -> begin
                    res = ""
                    for i in range(2, length = length(x) - 3)
                        res =
                            res *
                            "PauliVec[3]" *
                            "[" *
                            string(parse(Int, x[i]) + 1) *
                            "]" *
                            "⊗"
                    end
                    res =
                        res *
                        "PauliVec[3]" *
                        "[" *
                        string(parse(Int, x[end-1]) + 1) *
                        "]"
                end,
        )
    else
        h_str = replace(
            h,
            r"[01]+" =>
                (x) -> begin
                    res = ""
                    for i in range(1, length = length(x) - 1)
                        res =
                            res *
                            "PauliVec[3]" *
                            "[" *
                            string(parse(Int, x[i]) + 1) *
                            "]" *
                            "⊗"
                    end
                    res =
                        res *
                        "PauliVec[3]" *
                        "[" *
                        string(parse(Int, x[end]) + 1) *
                        "]"
                end,
        )
    end
    res = eval(Meta.parse(h_str))
    if normal == true
        normalize(res)
    else
        res
    end
end

"""
$(SIGNATURES)

Return the creation operator truncated at `num_levels`.

# Examples
```julia-repl
julia> creation_operator(3)
3×3 LinearAlgebra.Bidiagonal{Float64, Vector{Float64}}:
 0.0   ⋅        ⋅ 
 1.0  0.0       ⋅
  ⋅   1.41421  0.0
```
"""
creation_operator(num_levels::Int) = Bidiagonal(zeros(num_levels), sqrt.(1:num_levels-1), :L)

"""
$(SIGNATURES)

Return the annihilation operator truncated at `num_levels`.

# Examples
```julia-repl
julia> annihilation_operator(3)
3×3 LinearAlgebra.Bidiagonal{Float64, Vector{Float64}}:
 0.0  1.0   ⋅
  ⋅   0.0  1.41421
  ⋅    ⋅   0.0
```
"""
annihilation_operator(num_levels::Int) = Bidiagonal(zeros(num_levels), sqrt.(1:num_levels-1), :U);