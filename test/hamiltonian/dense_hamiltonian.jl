using OpenQuantumBase, Test

H = build_example_hamiltonian(1)

@test size(H) == (2, 2)
@test H(0.5) == π * (σx + σz)
@test evaluate(H, 0.5) == (σx + σz) / 2
@test get_cache(H) ≈ π * (σx + σz)
@test !isconstant(H)

# update_cache method
C = similar(σz)
update_cache!(C, H, 10, 0.5)
@test C == -1im * π * (σx + σz)

# update_vectorized_cache method
C = get_cache(H)
C = C⊗C
update_vectorized_cache!(C, H, 10, 0.5)
temp = -1im * π * (σx + σz)
@test C == σi ⊗ temp - transpose(temp) ⊗ σi

# in-place update for matrices
du = [1.0 + 0.0im 0; 0 0]
ρ  = PauliVec[1][1] * PauliVec[1][1]'
H(du, ρ, 2, 0.5)
@test du ≈ -1.0im * π * ((σx + σz) * ρ - ρ * (σx + σz))

# eigen-decomposition
w, v = eigen_decomp(H, 0.5)
@test w ≈ [-1, 1] / √2
@test abs(v[:, 1]'*[1-sqrt(2), 1] / sqrt(4-2*sqrt(2))) ≈ 1
@test abs(v[:, 2]'*[1+sqrt(2), 1] / sqrt(4+2*sqrt(2))) ≈ 1

Hrot= rotate(H, v)
@test evaluate(Hrot, 0.5) ≈ [-1 0; 0 1] / sqrt(2)

# error message test
@test_throws ArgumentError DenseHamiltonian([(s)->1-s, (s)->s], [σx, σz], unit=:hh)
