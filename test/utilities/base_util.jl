using OpenQuantumBase, Test

@test σx*PauliVec[1][1] == PauliVec[1][1]
@test σx*PauliVec[1][2] == -PauliVec[1][2]
@test σy*PauliVec[2][1] == PauliVec[2][1]
@test σy*PauliVec[2][2] == -PauliVec[2][2]
@test σz*PauliVec[3][1] == PauliVec[3][1]
@test σz*PauliVec[3][2] == -PauliVec[3][2]

@test bloch_to_state(π/2, 0.0) ≈ PauliVec[1][1]
@test bloch_to_state(0, 0) ≈ PauliVec[3][1]
@test bloch_to_state(π/2, π/2) ≈ PauliVec[2][1]
@test_throws ArgumentError bloch_to_state(2π, 0)
@test_throws ArgumentError bloch_to_state(π, 3π)

@test creation_operator(3) ≈ [0 0 0; 1 0 0; 0 sqrt(2) 0]
@test annihilation_operator(3) ≈ [0 1 0; 0 0 sqrt(2); 0 0 0]
@test number_operator(3) ≈ [0 0 0; 0 1 0; 0 0 2]

pauli_exp = "-0.1X1X2 + Z2"
res = OpenQuantumBase.split_pauli_expression(pauli_exp)
@test res[1] == ["-", "0.1", "X1X2"]
@test res[2] == ["+", "", "Z2"]
pauli_exp = "Y2X1-2Z1"
res = OpenQuantumBase.split_pauli_expression(pauli_exp)
@test res[1] == ["", "", "Y2X1"]
@test res[2] == ["-", "2", "Z1"]
pauli_exp = "X1X2 + 1.0imZ1"
res = OpenQuantumBase.split_pauli_expression(pauli_exp)
@test res[2] == ["+", "1.0im", "Z1"]

