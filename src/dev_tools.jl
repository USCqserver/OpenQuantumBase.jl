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

function build_example_hamiltonian(num_qubits; sp=false)
    A = (s) -> (1 - s)
    B = (s) -> s
    if num_qubits == 1
        sp ? SparseHamiltonian([A, B], [spσx, spσz]) : DenseHamiltonian([A, B], [σx, σz])
    elseif num_qubits == 4
        Hd = -standard_driver(4, sp = sp)
        Hp = -q_translate("ZIII+IZII+IIZI+IIIZ", sp = sp)
        sp ? SparseHamiltonian([A, B], [Hd, Hp]) : DenseHamiltonian([A, B], [Hd, Hp])
    end
end

"""
$(SIGNATURES)

Find the unique gap values upto `sigdigits` number of significant digits.
"""
function find_unique_gap(w::Vector{T}; sigdigits::Integer = 8) where T<:Real
    w = round.(w, sigdigits=sigdigits)
    w_matrix = w' .- w
    uniq_w = unique(w_matrix)
    uniq_w = sort(uniq_w[findall((x)->x>0, uniq_w)])
	indices = []
    for uw in uniq_w
        push!(indices, findall((x)->x==uw, w_matrix))
    end
	uniq_w, indices, findall((x)->x==0, w_matrix)
end