# # Constant Hamiltonian Interface
using OpenQuantumBase, Test, LinearAlgebra

# In HOQST a constant Hamiltonian can be constructed using the `ConstantHamiltonian` interface:
H₁ = ConstantHamiltonian(σx, unit=:ħ)
H₂ = ConstantHamiltonian(σx, static=false)
H₃ = ConstantHamiltonian(spσx⊗spσx)

# or `Hamiltonian` interface
Hₛ₁ = Hamiltonian(σx, unit=:ħ)
Hₛ₃ = Hamiltonian(spσx⊗spσx)

cache₁ = H₁ |> get_cache |> similar
cache₂ = H₂ |> get_cache |> similar
cache₃ = H₃ |> get_cache |> similar

@test H₁(2)==Hₛ₁(2)==σx
@test H₂(1)==2π*σx
@test H₃(0.2)==Hₛ₃(0.2)==2π*(σx⊗σx)


update_cache!(cache₁, H₁, nothing, 0.2)
update_cache!(cache₂, H₂, nothing, 0.1)
update_cache!(cache₃, H₃, nothing, 0.5)
@test cache₁ == -1.0im*σx
@test cache₂ == -2π*1im*σx
@test cache₃ == -2π*1im*(σx⊗σx)

vcache₁ = cache₁⊗cache₁ |> similar
vcache₂ = cache₂⊗cache₂ |> similar
vcache₃ = cache₃⊗spσi⊗spσi + spσi⊗spσi⊗cache₃ |> similar
update_vectorized_cache!(vcache₁, H₁, nothing, 0.2)
update_vectorized_cache!(vcache₂, H₂, nothing, 0.1)
update_vectorized_cache!(vcache₃, H₃, nothing, 0.5)
@test vcache₁ == -1.0im * OpenQuantumBase.vectorized_commutator(H₁(0.1))
@test vcache₂ == -1.0im * OpenQuantumBase.vectorized_commutator(H₂(1))
@test vcache₃ == -1.0im * OpenQuantumBase.vectorized_commutator(H₃(2))

ρ₁ = ones(ComplexF64, 2, 2)/2
ρ₃ = ones(ComplexF64, 4, 4)/4
H₁(cache₁, ρ₁, nothing, 0)
H₂(cache₂, ρ₁, nothing, 0.5)
H₃(cache₃, ρ₃, nothing, 1)
@test cache₁ == -1.0im*(H₁(0)*ρ₁-ρ₁*H₁(0))
@test cache₂ == -1.0im*(H₂(0.5)*ρ₁-ρ₁*H₂(0.5))
@test cache₃ == -1.0im*(H₃(1)*ρ₃-ρ₃*H₃(1))

we₁ = [-1, 1]/2/π
we₂ = [-1, 1]
we₃ = [-1, -1, 1, 1]
w₁, v₁ = eigen_decomp(H₁, 0)
w₂, v₂ = eigen_decomp(H₂, 0.5)
w₃, v₃= eigen_decomp(H₃, 1, lvl=4)

@test we₁ ≈ w₁
@test we₂ ≈ w₂
@test we₃ ≈ w₃

@test H₁(0) ≈ 2π * v₁'*Diagonal(w₁)*v₁
@test H₂(0.5) ≈ 2π * v₂'*Diagonal(w₂)*v₂

Hr₁ = rotate(H₁, v₁)
Hr₂ = rotate(H₂, v₂)
Hr₃ = rotate(H₃, v₃)

@test Hr₁(0) ≈2π*Diagonal(w₁)
@test Hr₂(0.5) ≈ 2π*Diagonal(w₂)
@test Hr₃(1) ≈ 2π*Diagonal(w₃)