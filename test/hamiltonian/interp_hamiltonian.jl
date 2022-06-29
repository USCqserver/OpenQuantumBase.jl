using OpenQuantumBase, Test

A = (s)->(1 - s)
B = (s)->s
u = [1.0 + 0.0im, 1] / sqrt(2)
ρ = PauliVec[1][1] * PauliVec[1][1]'

H = build_example_hamiltonian(1)
t_axis = range(0, 1, length=10)
H_list = [evaluate(H, t) for t in t_axis]

H_interp = InterpDenseHamiltonian(t_axis, H_list)
@test H_interp(0.5) == H(0.5)
@test !isconstant(H_interp)

# update_cache method
C = get_cache(H_interp, false)
update_cache!(C, H_interp, 10, 0.5)
@test C ≈ -1im * π * (σx + σz)

# update_vectorized_cache method
C = get_cache(H_interp, true)
update_vectorized_cache!(C, H, 10, 0.5)
temp = -1im * π * (σx + σz)
@test C == σi⊗temp - transpose(temp)⊗σi

# in-place update for matrices
du = [1.0 + 0.0im 0; 0 0]
H_interp(du, ρ, 2.0, 0.5)
@test du ≈ -1.0im * π * ((σx + σz) * ρ - ρ * (σx + σz))

H = build_example_hamiltonian(1, sp=true)
t_axis = range(0, 1, length=10)
H_list = [evaluate(H, t) for t in t_axis]
H_interp = InterpSparseHamiltonian(t_axis, H_list)
@test H_interp(0.5) == H(0.5)

# update_cache method
C = get_cache(H_interp, false)
update_cache!(C, H_interp, 10, 0.5)
@test C ≈ -1im * π * (spσx + spσz)

# update_vectorized_cache method
C = get_cache(H_interp, true)
update_vectorized_cache!(C, H, 10, 0.5)
temp = -1im * π * (spσx + spσz)
@test C == spσi⊗temp - transpose(temp)⊗spσi

du = [1.0 + 0.0im 0; 0 0]
H_interp(du, ρ, 1.0, 0.5)
@test du ≈ -1.0im * π * ((σx + σz) * ρ - ρ * (σx + σz))
