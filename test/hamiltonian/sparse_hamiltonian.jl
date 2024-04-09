using OpenQuantumBase, Test
import SparseArrays: spzeros
import LinearAlgebra: Diagonal, I

A = (s) -> (1 - s)
B = (s) -> s
u = [1.0 + 0.0im, 1] / sqrt(2)
ρ = u * u'

H_sparse = SparseHamiltonian([A, B], [spσx, spσz])
@test isdimensionlesstime(H_sparse)

Hs1 = SparseHamiltonian([A], [spσx])
Hs2 = SparseHamiltonian([B], [spσz])
Hs3 = Hs1 + Hs2
@test Hs3.m == H_sparse.m
@test Hs3.f == H_sparse.f

H_real = convert(Real, H_sparse)
@test eltype(H_real) <: Real
@test H_sparse(0.0) ≈ H_real(0.0)
@test !isconstant(H_sparse)

@test size(H_sparse) == (2, 2)
@test issparse(H_sparse)
@test H_sparse(0) ≈ 2π * spσx
@test evaluate(H_sparse, 0) == spσx
@test H_sparse(0.5) ≈ π * (spσx + spσz)
@test evaluate(H_sparse, 0.5) == (spσx + spσz) / 2
@test get_cache(H_sparse) ≈ π * (spσx + spσz)

du = [1.0+0.0im 0; 0 0]
H_sparse(du, ρ, 1.0, 0.5)
@test du ≈ -1.0im * π * ((σx + σz) * ρ - ρ * (σx + σz))

# update_cache method
C = similar(spσz)
update_cache!(C, H_sparse, 10, 0.5)
@test C == -1im * π * (spσx + spσz)

# update_vectorized_cache method
C = C ⊗ C
update_vectorized_cache!(C, H_sparse, 10, 0.5)
temp = -1im * π * (spσx + spσz)
@test C == spσi ⊗ temp - transpose(temp) ⊗ spσi

H_sparse = SparseHamiltonian([A, B], real.([spσx ⊗ spσi + spσi ⊗ spσx, 0.1spσz ⊗ spσi - spσz ⊗ spσz]))
w, v = eigen_decomp(H_sparse, 1.0)
vf = [0, 0, 0, 1.0]

@test w ≈ [-1.1, -0.9]
@test abs(v[end, 1]) ≈ 1
@test abs(v[1, 2]) ≈ 1

# ## Test suite for eigen decomposition of `SparseHamiltonian`
np = 5
Hd = standard_driver(np, sp=true)
Hp = alt_sec_chain(1, 0.5, 1, np, sp=true)
H_sparse = SparseHamiltonian([A, B], [Hd, Hp], unit=:ħ)
w1, v1 = haml_eigs(H_sparse, 0.5, nothing)
wl, vl = haml_eigs(H_sparse, 0.5, 3)
@test w1[1:3] ≈ wl
@test abs.(v1[:, 1:3]' * vl) |> Diagonal ≈ I

w2, v2 = haml_eigs(H_sparse, 0.5, 3, lobpcg=false)
@test w1 ≈ w2
@test v1 ≈ v2