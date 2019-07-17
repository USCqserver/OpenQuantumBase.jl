using QTBase, Test

A = (s)->(1-s)
B = (s)->s
u = [1.0+0.0im, 1]/sqrt(2)
ρ = u*u'

H = DenseHamiltonian([A, B], [σx, σz])

@test H(0) ≈ σx
@test H(0.5) ≈ 0.5σx + 0.5σz

du = [1.0+0.0im 0; 0 0]
H(du, ρ, 1.0, 0.5)
@test du ≈ -1.0im*((0.5σx + 0.5σz)*ρ -ρ*(0.5σx + 0.5σz)) + [1.0+0.0im 0; 0 0]

w, v = eigen_decomp(H, 0.5)
@test w ≈ [-1, 1]/sqrt(2)
w, v = eigen_decomp(H, 0.0)
@test w ≈ [-1, 1]
@test v ≈ [-1 1; 1 1]/sqrt(2)

H_new = p_copy(H)
@test H_new.u_cache == zeros(eltype(H_new.u_cache), size(H_new.u_cache))
