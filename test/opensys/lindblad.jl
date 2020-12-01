using OpenQuantumBase, Test, Random

coupling = ConstantCouplings(["Z"], unit=:ħ)
jfun(t₁, t₂) = 1.0
jfun(τ) = 1.0
# TODO: add test for unitary using StaticArrays
# const Sx = SMatrix{2,2}(σx)
unitary(t) = exp(-1.0im * t * σx)
tf = 5.0
u0 = PauliVec[1][1]
ρ = u0 * u0'
kernels = [(((1, 1),), coupling, OpenQuantumBase.SingleFunctionMatrix(jfun))]

L = OpenQuantumBase.quadgk((x) -> unitary(x)' * σz * unitary(x), 0, 5)[1]
ulind = OpenQuantumBase.ULELiouvillian(kernels, unitary, tf, 1e-8, 1e-6)
p = ODEParams(nothing, 5.0, (tf, t) -> t / tf)
dρ = zero(ρ)
ulind(dρ, ρ, p, 5.0)
@test dρ ≈ L * ρ * L' - 0.5 * (L' * L * ρ + ρ * L' * L) atol = 1e-6 rtol = 1e-6

# test for EᵨEnsemble
u0 = EᵨEnsemble([0.5, 0.5], [PauliVec[3][1], PauliVec[3][2]])
Random.seed!(1234)
@test [sample_state_vector(u0) for i in 1:4] == [[0.0 + 0.0im, 1.0 + 0.0im], [0.0 + 0.0im, 1.0 + 0.0im], [0.0 + 0.0im, 1.0 + 0.0im], [1.0 + 0.0im, 0.0 + 0.0im]]

Lz = Lindblad((s) -> 0.5, (s) -> σz)
Lx = Lindblad((s) -> 0.2, (s) -> σx)
Ł = OpenQuantumBase.lindblad_from_interactions(InteractionSet(Lz))[1]
p = ODEParams(nothing, 5.0, (tf, t) -> t / tf)
dρ = zero(ρ)
Ł(dρ, ρ, p, 5.0)
@test dρ ≈ [0 -0.5; -0.5 0]

cache = zeros(2, 2)
update_cache!(cache, Ł, p, 5.0)
@test cache == -0.25 * σz' * σz


Ł = OpenQuantumBase.lindblad_from_interactions(InteractionSet(Lz, Lx))[1]
p = ODEParams(nothing, 5.0, (tf, t) -> t / tf)
dρ = zero(ρ)
Ł(dρ, PauliVec[2][1] * PauliVec[2][1]', p, 5.0)
@test dρ ≈ [0 0.7im; -0.7im 0]

cache = zeros(2, 2)
update_cache!(cache, Ł, p, 5.0)
@test cache == -0.25 * σz' * σz - 0.1 * σx' * σx

Random.seed!(1234)
sample_res = [OpenQuantumBase.lind_jump(Ł, PauliVec[1][1], p, 0.5) for i in 1:4]
@test sample_res == [[1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im -1.0 + 0.0im],
[0.0 + 0.0im 1.0 + 0.0im; 1.0 + 0.0im 0.0 + 0.0im],
[1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im -1.0 + 0.0im],
[1.0 + 0.0im 0.0 + 0.0im; 0.0 + 0.0im -1.0 + 0.0im]]