using QTBase, Test

coupling = ConstantCouplings(["Z"], unit = :ħ)
cfun(t) = 1.0
unitary(t) = exp(-5.0im * t * σx)

redfield = Redfield(coupling, unitary, cfun)

u0 = PauliVec[1][1]
ρ = u0 * u0'
dρ = zero(ρ)
redfield(dρ, ρ, 5.0, 0.5)

Λ = QTBase.quadgk((x) -> unitary(x)' * σz * unitary(x), 0, 0.5)[1]
@test dρ ≈ -25 * (σz * (Λ * ρ - ρ * Λ') - (Λ * ρ - ρ * Λ') * σz) atol = 1e-6 rtol = 1e-6

A = zero(ρ ⊗ σi)
update_vectorized_cache!(A, redfield, 5.0, 0.5)
@test A * ρ[:] ≈ -25 * (σz*(Λ*ρ-ρ*Λ')-(Λ*ρ-ρ*Λ')*σz)[:]


dρ = zero(ρ)
redfield(dρ, ρ, UnitTime(5.0), 2.5)
Λ = QTBase.quadgk((x) -> unitary(x)' * σz * unitary(x), 0, 2.5)[1]
@test dρ ≈ -(σz * (Λ * ρ - ρ * Λ') - (Λ * ρ - ρ * Λ') * σz) atol = 1e-6 rtol = 1e-6
