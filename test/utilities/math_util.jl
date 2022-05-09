using OpenQuantumBase, Test

# === matrix decomposition
v = 1.0 * σx + 2.0 * σy + 3.0 * σz
res = matrix_decompose(v, [σx, σy, σz])
@test isapprox(res, [1.0, 2.0, 3.0])
# === positivity test ===
r = rand(2)
m =
    r[1] * PauliVec[1][2] * PauliVec[1][2]' +
    r[2] * PauliVec[1][1] * PauliVec[1][1]'
@test check_positivity(m)
@test !check_positivity(σx)
@test_logs (:warn, "Input fails the numerical test for Hermitian matrix. Use the upper triangle to construct a new Hermitian matrix.") check_positivity(σ₊)
# == units conversion test ===
@test temperature_2_freq(1e3) ≈ 20.8366176361328 atol = 1e-4 rtol = 1e-4
@test freq_2_temperature(20) ≈ 959.8489324422699 atol = 1e-4 rtol = 1e-4
@test temperature_2_freq(1e3) ≈ 1 /  temperature_2_β(1e3) / 2 / π
@test β_2_temperature(0.47) ≈ 16.251564065921915
# === unitary test ===
u_res = exp(-1.0im * 5 * 0.5 * σx)
@test check_unitary(u_res)
@test !check_unitary([0 1; 0 0])
# === integration test ===
@test OpenQuantumBase.cpvagk((x) -> 1.0, 0, -1, 1)[1] == 0

# == Hamiltonian analysis ===
DH = DenseHamiltonian([(s) -> 1 - s, (s) -> s], [σx, σz], unit=:ħ)
# hfun(s) = (1-s)*real(σx)+ s*real(σz)
# dhfun(s) = -real(σx) + real(σz)
t = [0.0, 1.0]
states = [PauliVec[1][2], PauliVec[1][1]]
res = inst_population(t, states, DH, lvl=1:2)
@test isapprox(res, [[1.0, 0], [0.5, 0.5]])

SH = SparseHamiltonian(
    [(s) -> -(1 - s), (s) -> s],
    [standard_driver(2, sp=true), (0.1 * spσz ⊗ spσi + spσz ⊗ spσz)],
    unit=:ħ,
)
H_check = DenseHamiltonian(
    [(s) -> -(1 - s), (s) -> s],
    [standard_driver(2), (0.1 * σz ⊗ σi + σz ⊗ σz)],
    unit=:ħ,
)
# spdhfun(s) =
#    real(standard_driver(2, sp = true) + (0.1 * spσz ⊗ spσi + spσz ⊗ spσz))
interaction = [spσz ⊗ spσi, spσi ⊗ spσz]
spw, spv = eigen_decomp(SH, [0.5])
w, v = eigen_decomp(H_check, [0.5])
@test w ≈ spw atol = 1e-4
@test isapprox(spv[:, 1, 1], v[:, 1, 1], atol=1e-4) ||
      isapprox(spv[:, 1, 1], -v[:, 1, 1], atol=1e-4)
@test isapprox(spv[:, 2, 1], v[:, 2, 1], atol=1e-4) ||
      isapprox(spv[:, 2, 1], -v[:, 2, 1], atol=1e-4)
# == utility math functions ==
@test log_uniform(1, 10, 3) == [1, 10^0.5, 10]

v = sqrt.([0.4, 0.6])
ρ1 = v*v'
ρ2 = [0.5 0; 0 0.5]
ρ3 = ones(3, 3)/3
@test ρ1 == partial_trace(ρ1 ⊗ ρ2 ⊗ ρ2, [1])
@test ρ2 == partial_trace(ρ1 ⊗ ρ2 ⊗ ρ2, [2])
@test ρ3 ≈ partial_trace(ρ1⊗ρ2⊗ρ3, [2,2,3], [3])
@test_throws ArgumentError partial_trace(ρ1⊗ρ2⊗ρ3, [3,2,3], [3])
@test_throws ArgumentError partial_trace(rand(4,5), [2,2], [1])
@test purity(ρ1) ≈ 1
@test purity(ρ2) == 0.5
@test check_pure_state(ρ1)
@test !check_pure_state(ρ2)
@test !check_pure_state([0.4 0.5; 0.5 0.6])

ρ = PauliVec[1][1]*PauliVec[1][1]'
σ = PauliVec[3][1]*PauliVec[3][1]'
@test fidelity(ρ, σ) ≈ 0.5
@test fidelity(ρ, ρ) ≈ 1
@test check_density_matrix(ρ)
@test !check_density_matrix(σx)

w = [1,1,1,2,3,4,4,5]
@test OpenQuantumBase.find_degenerate(w) == [[1,2,3],[6,7]]
@test isempty(OpenQuantumBase.find_degenerate([1,2,3]))

gibbs = gibbs_state(σz, 12)
@test gibbs[2,2] ≈ 1/(1+exp(-temperature_2_β(12)*2))
@test gibbs[1,1] ≈ 1 - 1/(1+exp(-temperature_2_β(12)*2))

@test low_level_matrix(σz⊗σz+0.1σz⊗σi, 2) == [0.0+0.0im   0.0+0.0im   0.0+0.0im  0.0+0.0im
0.0+0.0im  -0.9+0.0im   0.0+0.0im  0.0+0.0im
0.0+0.0im   0.0+0.0im  -1.1+0.0im  0.0+0.0im
0.0+0.0im   0.0+0.0im   0.0+0.0im  0.0+0.0im]
@test_logs (:warn, "Subspace dimension bigger than total dimension.") low_level_matrix(σz⊗σz+0.1σz⊗σi, 5)

@test !OpenQuantumBase.lesssim(1e-6, 2e-6, atol=1e-5)
@test OpenQuantumBase.lesssim(1e-6, 2e-6)