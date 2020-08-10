using QTBase, Test

coupling = ConstantCouplings(["Z"], unit = :ħ)
cfun(t) = 1.0
#TODO: add test for unitary using StaticArrays
#const Sx = SMatrix{2,2}(σx)
unitary(t) = exp(-1.0im * t * σx)
tf = 5.0
u0 = PauliVec[1][1]
ρ = u0 * u0'
redfield = DiagRedfieldGenerator(coupling, unitary, cfun, tf)
p = ODEParams(nothing, 5.0, (tf, t) -> t / tf)

Λ = QTBase.quadgk((x) -> unitary(x)' * σz * unitary(x), 0, 5)[1]
dρ = zero(ρ)
redfield(dρ, ρ, p, 5.0)
@test dρ ≈ -(σz * (Λ * ρ - ρ * Λ') - (Λ * ρ - ρ * Λ') * σz) atol = 1e-6 rtol =
    1e-6

A = zero(ρ ⊗ σi)
update_vectorized_cache!(A, redfield, p, 5.0)
@test A * ρ[:] ≈ -(σz*(Λ*ρ-ρ*Λ')-(Λ*ρ-ρ*Λ')*σz)[:]


# Λ = QTBase.quadgk((x) -> unitary(x)' * σz * unitary(x), 0, 2.5)[1]
# dρ = zero(ρ)
# redfield(dρ, ρ, p, 2.5)
# @test dρ ≈ -(σz * (Λ * ρ - ρ * Λ') - (Λ * ρ - ρ * Λ') * σz) atol = 1e-6 rtol =
#     1e-6
#
# A = zero(ρ ⊗ σi)
# update_vectorized_cache!(A, redfield, UnitTime(5.0), 2.5)
# @test A * ρ[:] ≈ -(σz*(Λ*ρ-ρ*Λ')-(Λ*ρ-ρ*Λ')*σz)[:]

# test for CustomCouplings
coupling = CustomCouplings([(s) -> σz], unit = :ħ)
redfield = DiagRedfieldGenerator(coupling, unitary, cfun, tf)

A = zero(ρ ⊗ σi)
Λ = QTBase.quadgk((x) -> unitary(x)' * σz * unitary(x), 0, 2.5)[1]
update_vectorized_cache!(A, redfield, p, 2.5)
@test A * ρ[:] ≈ -(σz*(Λ*ρ-ρ*Λ')-(Λ*ρ-ρ*Λ')*σz)[:]


# =============== CGME Test ===================
cgop = QTBase.CGGenerator(coupling, unitary, cfun, 1)
dρ = zero(ρ)
function integrand(x)
    a1 = unitary(x[1])' * σz * unitary(x[1])
    a2 = unitary(x[2])' * σz * unitary(x[2])
    a1 * ρ * a2 - 0.5 * (a2 * a1 * ρ + ρ * a2 * a1)
end
exp_res, err = QTBase.hcubature(integrand, [-0.5, -0.5], [0.5, 0.5])
cgop(dρ, ρ, p, 2.5)
@test dρ ≈ exp_res
