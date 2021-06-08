using OpenQuantumBase, LinearAlgebra, Test

num_qubits = 3
H₃ = random_ising(num_qubits)
@test size(H₃) == (2^num_qubits, 2^num_qubits)
@test isdiag(H₃)
@test !issparse(H₃)
@test issparse(random_ising(num_qubits, sp=true))
