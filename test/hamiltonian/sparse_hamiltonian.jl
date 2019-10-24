using QTBase, Test
import SparseArrays:spzeros

A = (s)->(1 - s)
B = (s)->s
u = [1.0 + 0.0im, 1] / sqrt(2)
ρ = u * u'

H_sparse = SparseHamiltonian([A, B], [spσx, spσz])

@test size(H_sparse) == (2,2)
@test is_sparse(H_sparse)
@test H_sparse(0) ≈ 2π * spσx
@test evaluate(H_sparse, 0) == spσx
@test H_sparse(0.5) ≈ π * (spσx + spσz)
@test evaluate(H_sparse, 0.5) == (spσx + spσz)/2

du = [1.0 + 0.0im 0; 0 0]
H_sparse(du, ρ, 1.0, 0.5)
@test du ≈ -1.0im * π * ((σx + σz) * ρ - ρ * (σx + σz))

H_new = p_copy(H_sparse)
@test H_new.u_cache == spzeros(eltype(H_sparse.u_cache), size(H_sparse.u_cache)...)

H_sparse = SparseHamiltonian([A, B], real.([spσx ⊗ spσi + spσi ⊗ spσx, 0.1spσz ⊗ spσi - spσz ⊗ spσz]))
w, v = eigen_decomp(H_sparse, 1.0)

@test w ≈ [-1.1, -0.9]
@test abs(v[end, 1]) ≈ 1
@test abs(v[1, 2]) ≈ 1

H_sparse(1.0)
w, v = QTBase.ode_eigen_decomp(H_sparse, 3)
@test w ≈ [-1.1, -0.9, 0.9] * 2π
