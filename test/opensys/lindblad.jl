using QTBase, Test, Random

coupling = ConstantCouplings(["Z"], unit=:ħ)
jfun(t₁, t₂) = 1.0
jfun(τ) = 1.0
# TODO: add test for unitary using StaticArrays
# const Sx = SMatrix{2,2}(σx)
unitary(t) = exp(-1.0im * t * σx)
tf = 5.0
u0 = PauliVec[1][1]
ρ = u0 * u0'
kernels = [(((1, 1),), coupling, QTBase.SingleCorrelation(jfun))]

L = QTBase.quadgk((x) -> unitary(x)' * σz * unitary(x), 0, 5)[1]
ulind = QTBase.ULindblad(kernels, unitary, tf, 1e-8, 1e-6)
p = ODEParams(nothing, 5.0, (tf, t) -> t / tf)
dρ = zero(ρ)
ulind(dρ, ρ, p, 5.0)
@test dρ ≈ L * ρ * L' - 0.5 * (L' * L * ρ + ρ * L' * L) atol = 1e-6 rtol = 1e-6

u0 = EᵨEnsemble([0.5, 0.5], [PauliVec[3][1], PauliVec[3][2]])
Random.seed!(1234)
@test [sample_state_vector(u0) for i in 1:4] == [[0.0 + 0.0im, 1.0 + 0.0im], [0.0 + 0.0im, 1.0 + 0.0im], [0.0 + 0.0im, 1.0 + 0.0im], [1.0 + 0.0im, 0.0 + 0.0im]]

#Lindblad((s)->0.5, (s)->σz)