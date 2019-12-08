using QTBase, Test

A = (s)->(1 - s)
B = (s)->s
u = [1.0 + 0.0im, 1] / sqrt(2)
ρ = u * u'

H = DenseHamiltonian([A, B], [σx, σz])
t_axis = range(0, 1, length=10)
H_list = [evaluate(H, t) for t in t_axis]

H_interp = InterpDenseHamiltonian(t_axis, H_list)
@test H_interp(0.5) == H(0.5)
@test H_interp(10, 0.5) == H(10, 0.5)
@test H_interp(UnitTime(10), 0.5) ≈ H(UnitTime(10), 0.5)


# update_cache method
C = get_cache(H_interp, false)
update_cache!(C, H_interp, 10, 0.5)
@test C ≈ -10im * π * (σx + σz)
update_cache!(C, H_interp, UnitTime(10), 5)
@test C ≈ -1im * π * (σx + σz)


# update_vectorized_cache method
C = get_cache(H_interp, true)
update_vectorized_cache!(C, H, 10, 0.5)
temp = -10im * π * (σx + σz)
@test C == σi⊗temp - transpose(temp)⊗σi
update_vectorized_cache!(C, H, UnitTime(10), 5)
temp = -1im * π * (σx + σz)
@test C == σi⊗temp - transpose(temp)⊗σi

# in-place update for matrices
du = [1.0 + 0.0im 0; 0 0]
H_interp(du, ρ, 2.0, 0.5)
@test du ≈ -2.0im * π * ((σx + σz) * ρ - ρ * (σx + σz))
H_interp(du, ρ, UnitTime(2), 1.0)
@test du ≈ -1.0im * π * ((σx + σz) * ρ - ρ * (σx + σz))


H = SparseHamiltonian([A, B], [spσx, spσz])
t_axis = range(0, 1, length=10)
H_list = [evaluate(H, t) for t in t_axis]
H_interp = InterpSparseHamiltonian(t_axis, H_list)
@test H_interp(0.5) == H(0.5)
@test H_interp(10, 0.5) == H(10, 0.5)
@test H_interp(UnitTime(10), 0.5) ≈ H(UnitTime(10), 0.5)

# update_cache method
C = get_cache(H_interp, false)
update_cache!(C, H_interp, 10, 0.5)
@test C ≈ -10im * π * (spσx + spσz)
update_cache!(C, H_interp, UnitTime(10), 5)
@test C ≈ -1im * π * (spσx + spσz)

# update_vectorized_cache method
C = get_cache(H_interp, true)
update_vectorized_cache!(C, H, 10, 0.5)
temp = -10im * π * (spσx + spσz)
@test C == spσi⊗temp - transpose(temp)⊗spσi
update_vectorized_cache!(C, H, UnitTime(10), 5)
temp = -1im * π * (spσx + spσz)
@test C == spσi⊗temp - transpose(temp)⊗spσi

du = [1.0 + 0.0im 0; 0 0]
H_interp(du, ρ, 1.0, 0.5)
@test du ≈ -1.0im * π * ((σx + σz) * ρ - ρ * (σx + σz))
