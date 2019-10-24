using QTBase, Test

A = (s)->(1 - s)
B = (s)->s
u = [1.0 + 0.0im, 1] / sqrt(2)
ρ = u * u'

H = DenseHamiltonian([A, B], [σx, σz])

@test size(H) == (2,2)
@test H(0) == 2π * σx
@test evaluate(H, 0) == σx
@test H(0.5) == π * (σx + σz)
@test evaluate(H, 0.5) == (σx + σz)/2

# Float/UnitTime on H(tf, t)
@test H(10, 0.5) == 10π * (σx + σz)
@test H(UnitTime(10.0), 5) == π * (σx + σz)

# in-place update for matrices
du = [1.0 + 0.0im 0; 0 0]
H(du, ρ, 2.0, 0.5)
@test du ≈ -2.0im * π * ((σx + σz) * ρ - ρ * (σx + σz))
H(du, ρ, UnitTime(2), 1.0)
@test du ≈ -1.0im * π * ((σx + σz) * ρ - ρ * (σx + σz))

# in-place update for vectors
du = [1.0 + 0.0im, 0]
H(du, u, 2.0, 0.5)
@test du ≈ -2.0im * π * (σx + σz) * u
H(du, u, UnitTime(2.0), 1.0)
@test du ≈ -1.0im * π * (σx + σz) * u


w, v = eigen_decomp(H, 0.5)
@test w ≈ [-1, 1] / sqrt(2)
w, v = eigen_decomp(H, 0.0)
@test w ≈ [-1, 1]
@test v ≈ [-1 1; 1 1] / sqrt(2)
H(0.5)
w, v = QTBase.ode_eigen_decomp(H, 2)
@test w ≈ [-1, 1] * sqrt(2) * π

H_new = p_copy(H)
@test H_new.u_cache != H.u_cache
