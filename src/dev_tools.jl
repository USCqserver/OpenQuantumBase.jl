"""
$(SIGNATURES)

Construct the matrix of a random 2D Ising lattice ``∑ᵢⱼJᵢⱼZᵢZⱼ``, where ``Jᵢⱼ``s are uniformly distributed between [-1, 1). Generate sparse matrix when `sp` is set to true.
"""
function random_ising(num_qubits::Integer; sp=false)
    J = Vector{Float64}()
    idx = Vector{Vector{Int64}}()
    for i ∈ 1:num_qubits
        for j ∈ i+1:num_qubits
            push!(J, 2*rand()-1)
            push!(idx, [i, j])
        end
    end
    two_local_term(J, idx, num_qubits, sp=sp)
end

"""
$(SIGNATURES)

Construct the matrix of the alternating sectors chain model described in (https://www.nature.com/articles/s41467-018-05239-9#Sec1). Generate sparse matrix when `sp` is set to true.
"""
function alt_sec_chain(w1, w2, n, num_qubits; sp=false)
    J = Vector{Float64}()
    idx = Vector{Vector{Int64}}()
    for i in 1:num_qubits-1
        push!(J, isodd(ceil(Int, i/n)) ? w1 : w2)
        push!(idx, [i, i+1])
    end
    two_local_term(J, idx, num_qubits, sp=sp)
end