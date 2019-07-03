using QTBase, Test

A = (s)->(1-s)
B = (s)->s
u = [1.0+0.0im, 1]/sqrt(2)
ρ = u*u'

H = hamiltonian_factory([A, B], [σx, σz])
H_sparse = hamiltonian_factory([A, B], [spσx, spσz])

@test H(0) ≈ σx
@test H_sparse(0) ≈ spσx
@test H(0.5) ≈ 0.5σx + 0.5σz
@test H_sparse(0.5) ≈ 0.5spσx + 0.5spσz

du = [1.0+0.0im 0; 0 0]
H(du, ρ, 0.5)
@test du ≈ -1.0im*((0.5σx + 0.5σz)*ρ -ρ*(0.5σx + 0.5σz)) + [1.0+0.0im 0; 0 0]

du = [1.0+0.0im 0; 0 0]
H_sparse(du, ρ, 0.5)
@test du ≈ -1.0im*((0.5σx + 0.5σz)*ρ -ρ*(0.5σx + 0.5σz)) + [1.0+0.0im 0; 0 0]

w, v = eigen_decomp(H, 0.5)
@test w ≈ [-1, 1]/sqrt(2)
w, v = eigen_decomp(H, 0.0)
@test w ≈ [-1, 1]
@test v ≈ [-1 1; 1 1]/sqrt(2)

scale!(H, 2)
@test H(0.5) ≈ σx + σz
scale!(H_sparse, 2)
@test H_sparse(0.5) ≈ spσx + spσz

H_sparse = hamiltonian_factory([A, B], [spσx⊗spσi + spσi⊗spσx, 0.1spσz⊗spσi-spσz⊗spσz], is_real=true)
w, v = eigen_decomp(H_sparse, 1.0)

@test w ≈ [-1.1, -0.9]
@test abs(v[end, 1]) ≈ 1
@test abs(v[1, 2]) ≈ 1
