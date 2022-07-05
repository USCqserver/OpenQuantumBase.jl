using OpenQuantumBase, Test

H₁ = ConstantHamiltonian(σx, unit=:ħ)
H₂ = ConstantHamiltonian(σx, static=false)
H₃ = ConstantHamiltonian(spσx⊗spσx)

cache₁ = H₁ |> get_cache |> similar
cache₂ = H₂ |> get_cache |> similar
cache₃ = H₃ |> get_cache |> similar

@test H₁(2)==σx
@test H₂(1) == 2π*σx
@test H₃(0.2) == 2π*(σx⊗σx)


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
H₂(cache₂, ρ₂, nothing, 0.5)
H₃(cache₃, ρ₃, nothing, 1)
@test cache₁ == -1.0im*(H₁(0)*ρ₁-ρ₁*H₁(0))
@test cache₂ == -1.0im*(H₂(0.5)*ρ₁-ρ₁*H₂(0.5))
@test cache₃ == -1.0im*(H₃(1)*ρ₃-ρ₃*H₃(1))