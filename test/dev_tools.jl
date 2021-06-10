using OpenQuantumBase, LinearAlgebra, Test

num_qubits = 3
H₃ = random_ising(num_qubits)
@test size(H₃) == (2^num_qubits, 2^num_qubits)
@test isdiag(H₃)
@test !issparse(H₃)
@test issparse(random_ising(num_qubits, sp=true))

@test alt_sec_chain(1,0.5,1,3) == σz⊗σz⊗σi + 0.5*σi⊗σz⊗σz
