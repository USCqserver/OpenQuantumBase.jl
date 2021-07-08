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