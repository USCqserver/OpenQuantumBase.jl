using QTBase, Test

coupling = ConstantCouplings(["Z"], unit = :ħ)
cfun(t) = 1.0
unitary(t) = exp(-5.0im * t * σx)
tf = 5.0
u0 = PauliVec[1][1]
ρ = u0 * u0'
redfield = Redfield(coupling, unitary, cfun, tf)

Λ = QTBase.quadgk((x) -> unitary(x)' * σz * unitary(x), 0, 0.5)[1]
dρ = zero(ρ)
redfield(dρ, ρ, 5.0, 0.5)
@test dρ ≈ -25 * (σz * (Λ * ρ - ρ * Λ') - (Λ * ρ - ρ * Λ') * σz) atol = 1e-6 rtol = 1e-6

A = zero(ρ ⊗ σi)
update_vectorized_cache!(A, redfield, 5.0, 0.5)
@test A * ρ[:] ≈ -25 * (σz*(Λ*ρ-ρ*Λ')-(Λ*ρ-ρ*Λ')*σz)[:]


Λ = QTBase.quadgk((x) -> unitary(x)' * σz * unitary(x), 0, 2.5)[1]
dρ = zero(ρ)
redfield(dρ, ρ, UnitTime(5.0), 2.5)
@test dρ ≈ -(σz * (Λ * ρ - ρ * Λ') - (Λ * ρ - ρ * Λ') * σz) atol = 1e-6 rtol =
    1e-6

A = zero(ρ ⊗ σi)
update_vectorized_cache!(A, redfield, UnitTime(5.0), 2.5)
@test A * ρ[:] ≈ -(σz*(Λ*ρ-ρ*Λ')-(Λ*ρ-ρ*Λ')*σz)[:]

# test for CustomCouplings
coupling = CustomCouplings([(s) -> σz], unit = :ħ)
redfield = Redfield(coupling, unitary, cfun, tf)

A = zero(ρ ⊗ σi)
Λ = QTBase.quadgk((x) -> unitary(x)' * σz * unitary(x), 0, 0.5)[1]
update_vectorized_cache!(A, redfield, 5.0, 0.5)
@test A * ρ[:] ≈ -25 * (σz*(Λ*ρ-ρ*Λ')-(Λ*ρ-ρ*Λ')*σz)[:]


A = zero(ρ ⊗ σi)
Λ = QTBase.quadgk((x) -> unitary(x)' * σz * unitary(x), 0, 2.5)[1]
update_vectorized_cache!(A, redfield, UnitTime(5.0), 2.5)
@test A * ρ[:] ≈ -(σz*(Λ*ρ-ρ*Λ')-(Λ*ρ-ρ*Λ')*σz)[:]


# =============== CGME Test ===================
cgop = CGOP(coupling, unitary, cfun, 1)
dρ = zero(ρ)
function integrand(x)
    a1 = unitary(x[1])' * σz * unitary(x[1])
    a2 = unitary(x[2])' * σz * unitary(x[2])
    25 * (a1 * ρ * a2 - 0.5 * (a2 * a1 * ρ + ρ * a2 * a1))
end
exp_res, err = QTBase.hcubature(integrand, [-0.1, -0.1], [0.1, 0.1])
cgop(dρ, ρ, 5.0, 0.5)
@test dρ ≈ exp_res * 5

cgop = CGOP(coupling, (x) -> unitary(x / 5), cfun, 1)
dρ = zero(ρ)
cgop(dρ, ρ, UnitTime(5.0), 2.5)
@test dρ ≈ exp_res
