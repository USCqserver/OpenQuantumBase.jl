using QTBase, Test

coupling = ConstantCouplings(["Z"], unit=:ħ)
cfun(t) = 1.0
unitary(t) = exp(-5.0im*t*σx)

redfield = Redfield(coupling, unitary, cfun)

u0 = PauliVec[1][1]
ρ = u0 * u0'
dρ = similar(ρ)
redfield(dρ, ρ, 5.0, 0.5)

Λ = QTBase.quadgk((x)-> unitary(x)' * σz * unitary(x), 0, 0.5)[1]
@test dρ ≈ -25*(σz*(Λ*ρ-ρ*Λ')-(Λ*ρ-ρ*Λ')*σz) atol=1e-6 rtol=1e-6
