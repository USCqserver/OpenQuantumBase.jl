using OpenQuantumBase, Test, Random
import LinearAlgebra: Diagonal, diag

# # Dense Hamiltonian AME tests
# ## Set up problems
# Set up mock functions for the bath spectrum and corresponding lambshift
gamma(x) = x >= 0 ? x + 1 : (1 - x) * exp(x)
lamb(x) = x + 0.1

# ## Two-qubit dense Hamiltonian
H = DenseHamiltonian(
    [(s) -> 1 - s, (s) -> s],
    [-standard_driver(2), (0.1 * σz ⊗ σi + 0.5 * σz ⊗ σz)]
)
coupling = ConstantCouplings(["ZI+IZ"])
davies = OpenQuantumBase.DaviesGenerator(coupling, gamma, lamb)
onesided = OpenQuantumBase.OneSidedAMELiouvillian(coupling, OpenQuantumBase.SingleFunctionMatrix(gamma), OpenQuantumBase.SingleFunctionMatrix(lamb), [(1, 1)])
ops = [2π * (σz ⊗ σi + σi ⊗ σz)]

w, v = eigen_decomp(H, 0.5, lvl=4)
w = 2π * w
ψ = (v[:, 1] + v[:, 2] + v[:, 3]) / sqrt(3)
ρ = ψ * ψ'
u = v' * ρ * v
g_idx = OpenQuantumBase.GapIndices(w, 8, 8)

# ## Tests
# Test density matrix update function for `DaviesGenerator` type
dρ = OpenQuantumBase.ame_update_test(ops, ρ, w, v, gamma, lamb)
du = zeros(ComplexF64, (4, 4))
davies(du, u, g_idx, v, 0.5)
@test v * du * v' ≈ dρ atol = 1e-6 rtol = 1e-6

onesided_dρ = OpenQuantumBase.onesided_ame_update_test(ops, ρ, w, v, gamma, lamb)
du = zeros(ComplexF64, (4, 4))
onesided(du, u, g_idx, v, 0.5)
@test v * du * v' ≈ onesided_dρ atol = 1e-6 rtol = 1e-6

onesided_dρ = OpenQuantumBase.onesided_ame_update_test([v * op * v' for op in ops], ρ, w, v, gamma, lamb)
du = zeros(ComplexF64, (4, 4))
onesided(du, u, g_idx, 0.5)
@test du ≈ v' * onesided_dρ * v atol = 1e-6 rtol = 1e-6

cache = zeros(ComplexF64, (4, 4))
exp_effective_H = OpenQuantumBase.ame_trajectory_Heff_test(ops, w, v, gamma, lamb)
update_cache!(cache, davies, g_idx, v, 0.5)
@test v * cache * v' ≈ -1.0im * exp_effective_H atol = 1e-6 rtol = 1e-6

# Test for dense Hamiltonian DiffEqLiouvillian
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

#= ================ Test for lindblad_jump ================= =#
#TODO: better test routine for `lindblad_jump`
jump_op = OpenQuantumBase.lindblad_jump(ame_op, ψ, p, 1)
@test size(jump_op) == (4, 4)

# ## Two-qubit constant Hamiltonian
Hc = Hamiltonian(standard_driver(2) + alt_sec_chain(1, 0.5, 1, 2))
wc, vc = eigen_decomp(Hc, lvl=2^2)
wc = 2π*wc
ψc = (vc[:, 1] + vc[:, 2] + vc[:, 3]) / sqrt(3)
ρc = ψc * ψc'
uc = vc' * ρc * vc
couplings_const = collective_coupling("Z", 2)
gapidx_const = OpenQuantumBase.build_gap_indices(wc, 8, 8, Inf, 4)
davies_const = OpenQuantumBase.build_const_davies(rotate(couplings_const, vc), gapidx_const, gamma, lamb)

# Test density matrix update function for `ConstDaviesGenerator` type
dρ = OpenQuantumBase.ame_update_test(couplings_const(0), ρc, wc, vc, gamma, lamb)
du = zeros(ComplexF64, (4, 4))
davies_const(du, uc, nothing, 0)
@test dρ ≈ vc * du * vc'

effective_H = OpenQuantumBase.ame_trajectory_Heff_test(couplings_const(0), wc, vc, gamma, lamb)
cache = zeros(ComplexF64, (4, 4))
update_cache!(cache, davies_const, nothing, 0)
@test -1.0im * effective_H ≈	vc * cache * vc'

# Test for lindblad_jump
Lc = OpenQuantumBase.build_diffeq_liouvillian(Hamiltonian(OpenQuantumBase.sparse(OpenQuantumBase.Diagonal(wc)), unit=:ħ), [], [davies_const] , 4)

jump_op = OpenQuantumBase.lindblad_jump(Lc, vc'*ψc, p, 1)
@test size(jump_op) == (4, 4)


#= === Test for ense Hamiltonian with size smaller than truncation levels === =#
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

#= ===== Tests for sparse Hamiltonians ===== =#
#= ===== Problem set up ===== =#
Hd = standard_driver(4; sp=true)
Hp = q_translate("-0.9ZZII+IZZI-0.9IIZZ"; sp=true)
H = SparseHamiltonian([(s) -> 1 - s, (s) -> s], [Hd, Hp])
ops = [2π * q_translate("ZIII+IZII+IIZI+IIIZ")]
coupling = ConstantCouplings(["ZIII+IZII+IIZI+IIIZ"])
davies = OpenQuantumBase.DaviesGenerator(coupling, gamma, lamb)
w, v = eigen_decomp(H, 0.5, lvl=4, lobpcg=false)
w = 2π * real(w)
g_idx = OpenQuantumBase.GapIndices(w, 8, 8)

ψ = (v[:, 1] + v[:, 2] + v[:, 3]) / sqrt(3)
ρ = ψ * ψ'
u = v' * ρ * v
dρ = OpenQuantumBase.ame_update_test(ops, ρ, w, v, gamma, lamb)
exp_effective_H =
    OpenQuantumBase.ame_trajectory_Heff_test(ops, w, v, gamma, lamb) + v * Diagonal(w) * v'

#= ============ Tests ============ =#
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

#= ===== Tests for adiabatc frame Hamiltonians ===== =#
#= ===== Problem set up ===== =#
H = AdiabaticFrameHamiltonian((s) -> [0, s, 1 - s, 1], nothing)
hmat = H(2.0, 0.4)
w = diag(hmat)
v = collect(Diagonal(ones(4)))
coupling = CustomCouplings([(s) -> s * (σx ⊗ σi + σi ⊗ σx) + (1 - s) * (σz ⊗ σi + σi ⊗ σz)])
davies = OpenQuantumBase.DaviesGenerator(coupling, gamma, lamb)

ψ = (v[:, 1] + v[:, 2] + v[:, 3]) / sqrt(3)
ρ = ψ * ψ'

#= ============ Tests ============ =#
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
gfun = (w) -> w >= 0 ? 1.0 : exp(-0.5)
cbath = CorrelatedBath(((1, 2), (2, 1)), spectrum=[(w)->0 gfun; gfun (w)->0])
D = OpenQuantumBase.davies_from_interactions(InteractionSet(Interaction(coupling, cbath)), 1:10, false, Dict())[1]
@test typeof(D) <: OpenQuantumBase.CorrelatedDaviesGenerator
du = zeros(ComplexF64, 2, 2)
ρ = [0.5 0; 0 0.5]
ω = [1, 2]
g_idx = OpenQuantumBase.GapIndices(ω, 8, 8)
D(du, ρ, g_idx, 0.5)
@test du ≈ zeros(2, 2)

coupling = ConstantCouplings([σ₋, σ₋], unit=:ħ)
D = OpenQuantumBase.davies_from_interactions(InteractionSet(Interaction(coupling, cbath)), 1:10, false, Dict())[1]
@test typeof(D) <: OpenQuantumBase.CorrelatedDaviesGenerator
du = zeros(ComplexF64, 2, 2)
ρ = [0.5 0.5; 0.5 0.5]
ω = [1, 2]
g_idx = OpenQuantumBase.GapIndices(ω, 8, 8)
D(du, ρ, g_idx, 0.5)
@test du ≈ [-exp(-0.5) -0.5*exp(-0.5); -0.5*exp(-0.5) exp(-0.5)]
