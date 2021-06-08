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