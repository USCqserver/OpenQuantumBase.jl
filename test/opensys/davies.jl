using OpenQuantumBase, Test, Random
import LinearAlgebra: Diagonal, diag

#= =============== Define  test functions ================= =#
# calculate the one sided AME term
function onesided_ame_udpate(op, ρ, w, v, γ, S)
    ω_ba = transpose(w) .- w
    Γ =  0.5 * γ.(ω_ba) + 1.0im * S.(ω_ba)
    L = v * (Γ .* (v' * op * v)) * v'
    K = L * ρ * op - op * L * ρ
    K + K'
end

gamma(x) = x >= 0 ? x + 1 : (1 - x) * exp(x)
lamb(x) = x + 0.1
#= ================ Define test functions ================= =#

H = DenseHamiltonian(
    [(s) -> 1 - s, (s) -> s],
    [-standard_driver(2), (0.1 * σz ⊗ σi + 0.5 * σz ⊗ σz)]
)
coupling = ConstantCouplings(["ZI+IZ"])
davies = OpenQuantumBase.DaviesGenerator(coupling, gamma, lamb)
onesided = OpenQuantumBase.OneSidedAMELiouvillian(coupling, OpenQuantumBase.SingleFunctionMatrix(γ), OpenQuantumBase.SingleFunctionMatrix(S), [(1, 1)])
ops = [2π * (σz ⊗ σi + σi ⊗ σz)]

w, v = eigen_decomp(H, 0.5, lvl=4)
w = 2π * w
ψ = (v[:, 1] + v[:, 2] + v[:, 3]) / sqrt(3)
ρ = ψ * ψ'
u = v' * ρ * v
w_ab = transpose(w) .- w
gm = gamma.(w_ab)
sm = lamb.(w_ab)
g_idx = OpenQuantumBase.GapIndices(w, 8, 8)

# expected result
dρ = OpenQuantumBase.ame_update_test(ops, ρ, w, v, gamma, lamb)

du = zeros(ComplexF64, (4, 4))
davies(du, u, g_idx, v, 0.5)
@test v * du * v' ≈ dρ atol = 1e-6 rtol = 1e-6

onesided_dρ = onesided_ame_udpate(op, ρ, w, v, γ, S) 
du = zeros(ComplexF64, (4, 4))
onesided(du, u, w_ab, v, 0.5)
@test v * du * v' ≈ onesided_dρ atol = 1e-6 rtol = 1e-6

onesided_dρ = onesided_ame_udpate(v*op*v', ρ, w, v, γ, S) 
du = zeros(ComplexF64, (4, 4))
onesided(du, u, w_ab, 0.5)
@test  du ≈ v' * onesided_dρ * v atol = 1e-6 rtol = 1e-6

cache = zeros(ComplexF64, (4, 4))
exp_effective_H = OpenQuantumBase.ame_trajectory_Heff_test(ops, w, v, gamma, lamb)
update_cache!(cache, davies, g_idx, v, 0.5)
@test v * cache * v' ≈ -1.0im * exp_effective_H atol = 1e-6 rtol = 1e-6

# test for DiffEqLiouvillian
ame_op = DiffEqLiouvillian(H, [davies], [], 4)
p = ODEParams(H, 2.0, (tf, t) -> t / tf)
exp_effective_H = OpenQuantumBase.ame_trajectory_Heff_test(ops, w, v, gamma, lamb) + H(0.5)
cache = zeros(ComplexF64, 4, 4)
update_cache!(cache, ame_op, p, 1)
@test cache ≈ -1.0im * exp_effective_H atol = 1e-6 rtol = 1e-6

hmat = H(0.5)
expected_drho = dρ - 1.0im * (hmat * ρ - ρ * hmat)
du = zeros(ComplexF64, (4, 4))
ame_op(du, ρ, p, 1)
@test du ≈ expected_drho atol = 1e-6 rtol = 1e-6

# best test routine for `lindblad_jump`
jump_op = OpenQuantumBase.lindblad_jump(ame_op, ψ, p, 1)
@test size(jump_op) == (4, 4)

# Dense Hamiltonian with size smaller than truncation levels
H = DenseHamiltonian(
    [(s) -> 1 - s, (s) -> s],
    [-standard_driver(4), random_ising(4)]
)

coupling = collective_coupling("Z", 4)
davies = OpenQuantumBase.DaviesGenerator(coupling, gamma, lamb)
ame_op = DiffEqLiouvillian(H, [davies], [], 20)
p = ODEParams(H, 2.0, (tf, t) -> t / tf)
cache = zeros(ComplexF64, 16, 16)
update_cache!(cache, ame_op, p, 1)
@test cache != zeros(ComplexF64, 16, 16)

# Sparse Hamiltonian test
Hd = standard_driver(4; sp=true)
Hp = q_translate("-0.9ZZII+IZZI-0.9IIZZ"; sp=true)
H = SparseHamiltonian([(s) -> 1 - s, (s) -> s], [Hd, Hp])
ops = [2π * q_translate("ZIII+IZII+IIZI+IIIZ")]
coupling = ConstantCouplings(["ZIII+IZII+IIZI+IIIZ"])
davies = OpenQuantumBase.DaviesGenerator(coupling, gamma, lamb)
w, v = eigen_decomp(H, 0.5, lvl=4)
w = 2π * real(w)
g_idx = GapIndices(w, 8, 8)

ψ = (v[:, 1] + v[:, 2] + v[:, 3]) / sqrt(3)
ρ = ψ * ψ'
u = v' * ρ * v
dρ = OpenQuantumBase.ame_update_test(ops, ρ, w, v, gamma, lamb)
exp_effective_H =
    OpenQuantumBase.ame_trajectory_Heff_test(ops, w, v, gamma, lamb) + v * Diagonal(w) * v'

ame_op = DiffEqLiouvillian(H, [davies], [], 4)
du = zeros(ComplexF64, (16, 16))
ame_op(du, ρ, p, 1)
hmat = H(0.5)
@test isapprox(
    du,
    dρ - 1.0im * (hmat * ρ - ρ * hmat),
    atol=1e-6,
    rtol=1e-6,
)

cache = zeros(ComplexF64, 16, 16)
update_cache!(cache, ame_op, p, 1)
@test cache ≈ -1im * exp_effective_H atol = 1e-6 rtol = 1e-6

# test for adiabatc frame Hamiltonian
H = AdiabaticFrameHamiltonian((s) -> [0, s, 1 - s, 1], nothing)
hmat =  H(2.0, 0.4)
w = diag(hmat)
v = collect(Diagonal(ones(4)))
coupling = CustomCouplings([(s) -> s * (σx ⊗ σi + σi ⊗ σx) + (1 - s) * (σz ⊗ σi + σi ⊗ σz)])
davies = OpenQuantumBase.DaviesGenerator(coupling, gamma, lamb)

ψ = (v[:, 1] + v[:, 2] + v[:, 3]) / sqrt(3)
ρ = ψ * ψ'

dρ = OpenQuantumBase.ame_update_test(coupling(0.4), ρ, w, v, gamma, lamb)
p = ODEParams(H, 2.0, (tf, t) -> t / tf)
ame_op = DiffEqLiouvillian(H, [davies], [], 4)
du = zeros(ComplexF64, (4, 4))
ame_op(du, ρ, p, 0.4 * 2)

@test isapprox(
    du,
    dρ - 1.0im * (hmat * ρ - ρ * hmat),
    atol=1e-6,
    rtol=1e-6,
)

# test suite for CorrelatedDaviesGenerator
coupling = ConstantCouplings([σ₊, σ₋], unit=:ħ)
γfun = (w) -> w >= 0 ? 1.0 : exp(-0.5)
cbath = CorrelatedBath(((1, 2), (2, 1)), spectrum=[(w) -> 0 γfun; γfun (w) -> 0])
D = OpenQuantumBase.davies_from_interactions(InteractionSet(Interaction(coupling, cbath)), 1:10, false, nothing)[1]
@test typeof(D) <: OpenQuantumBase.CorrelatedDaviesGenerator
du = zeros(ComplexF64, 2, 2)
ρ = [0.5 0;0 0.5]
ω = [1, 2]
ω =  ω' .- ω
D(du, ρ, ω, 0.5)
@test du ≈ [(1 - exp(-0.5)) * 0.5 0; 0 -(1 - exp(-0.5)) * 0.5]