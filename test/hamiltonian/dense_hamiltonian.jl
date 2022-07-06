using OpenQuantumBase, Test

A = (s) -> (1 - s)
B = (s) -> s
H = DenseHamiltonian([A, B], [σx, σz])
Hc = H |> get_cache

@test size(H) == (2, 2)
@test size(Hc) == size(H)
@test eltype(Hc) == eltype(H)
@test H(0.5) == π * (σx + σz)
@test evaluate(H, 0.5) == (σx + σz) / 2
@test !isconstant(H)
@test isdimensionlesstime(H)

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

Hnd = DenseHamiltonian([A, B], [σx, σz], dimensionless_time=false)
@test !isdimensionlesstime(Hnd)

# test for Hamiltonian interface
@test Hamiltonian([A, B], [σx, σz], static=false) |> typeof <: DenseHamiltonian

# test for Static Hamiltonian type 
Hst = Hamiltonian([A, B], [σx, σz], unit=:ħ)
Hstc = Hst |> get_cache

@test size(Hst) == (2, 2)
@test size(Hstc) == size(Hst)
@test eltype(Hstc) == eltype(Hst)
@test Hst(0.5) == (σx + σz) / 2
@test evaluate(Hst, 0.5) == (σx + σz) / 4 / π
@test !isconstant(Hst)
@test isdimensionlesstime(Hst)

# update_cache method
update_cache!(Hstc, Hst, 10, 0.5)
@test Hstc == -0.5im * (σx + σz)

# update_vectorized_cache method
C = Hstc⊗Hstc
update_vectorized_cache!(C, Hst, 10, 0.5)
temp = -0.5im * (σx + σz)
@test C == σi ⊗ temp - transpose(temp) ⊗ σi

# in-place update for matrices
du = [1.0 + 0.0im 0; 0 0]
ρ  = PauliVec[1][1] * PauliVec[1][1]'
Hst(du, ρ, 2, 0.5)
@test du ≈ -0.5im * ((σx + σz) * ρ - ρ * (σx + σz))

# eigen-decomposition
w, v = eigen_decomp(Hst, 0.5)
@test 2π*w ≈ [-1, 1] / √2
@test abs(v[:, 1]'*[1-sqrt(2), 1] / sqrt(4-2*sqrt(2))) ≈ 1
@test abs(v[:, 2]'*[1+sqrt(2), 1] / sqrt(4+2*sqrt(2))) ≈ 1

Hrot= rotate(Hst, v)
@test 2π*evaluate(Hrot, 0.5) ≈ [-1 0; 0 1] / sqrt(2)
