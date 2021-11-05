using OpenQuantumBase, Test, Random
import LinearAlgebra: Diagonal, diag


#= =============== Define  test functions ================= =#
# calculate the AME linblad term using the formula in reference paper
function ame_update_term(op, ρ, w, v, γ, S)
    L_ij = Array{Array{Complex{Float64},2},2}(undef, (4, 4))
    A_ij = Array{Complex{Float64},2}(undef, (4, 4))
    for i = 1:4
        for j = 1:4
            A_ij[i, j] = v[:, i]' * op * v[:, j]
            L_ij[i, j] = v[:, i]' * op * v[:, j] * v[:, i] * v[:, j]'
        end
    end
    dρ = []
    hls = []
    w_ab = Array{Float64,2}(undef, (4, 4))
    for i = 1:4
        for j = 1:4
            w_ab[i, j] = w[j] - w[i]
            if i != j
                T = L_ij[i, j]' * L_ij[i, j]
                push!(
                    dρ,
                    γ(w[j] - w[i]) * (
                        L_ij[i, j] * ρ * L_ij[i, j]' -
                        0.5 * (T * ρ + ρ * T)
                    ),
                )
                push!(hls, T * S(w[j] - w[i]))
            end
            T = L_ij[i, i]' * L_ij[j, j]
            push!(
                dρ,
                γ(0) *
                (L_ij[i, i] * ρ * L_ij[j, j]' - 0.5 * (T * ρ + ρ * T)),
            )
            push!(hls, T * S(0))
        end
    end
    dρ = sum(dρ)
    hls = sum(hls)
    dρ = dρ - 1.0im * (hls * ρ - ρ * hls)
    dρ, A_ij
end

# calculate the ame trajectory term
function ame_trajectory_update_term(op, w, v, γ, S)
    A_ij = Array{Complex{Float64},2}(undef, (4, 4))
    for i = 1:4
        for j = 1:4
            A_ij[i, j] = v[:, i]' * op * v[:, j]
        end
    end
    cache = zeros(ComplexF64, size(v, 1), size(v, 1))
    for i = 1:4
        for j = 1:4
            cache -=
                abs2(A_ij[i, j]) *
                (0.5 * γ(w[j] - w[i]) + 1.0im * S(w[j] - w[i])) *
                v[:, j] *
                v[:, j]'
        end
    end
    cache
end

# calculate the one sided AME term
function onesided_ame_udpate(op, ρ, w, v, γ, S)
    ω_ba = transpose(w) .- w
    Γ =  0.5 * γ.(ω_ba) + 1.0im * S.(ω_ba)
    L = v * (Γ .* (v' * op * v)) * v'
    K = L * ρ * op - op * L * ρ
    K + K'
end

γ(x) = x >= 0 ? x + 1 : (1 - x) * exp(x)
S(x) = x + 0.1
#= ================ Define test functions ================= =#

H = DenseHamiltonian(
    [(s) -> 1 - s, (s) -> s],
    [-standard_driver(2), (0.1 * σz ⊗ σi + 0.5 * σz ⊗ σz)]
)
coupling = ConstantCouplings(["ZI+IZ"])
davies = OpenQuantumBase.DaviesGenerator(coupling, γ, S)
onesided = OpenQuantumBase.OneSidedAMELiouvillian(coupling, OpenQuantumBase.SingleFunctionMatrix(γ), OpenQuantumBase.SingleFunctionMatrix(S), [(1, 1)])
op = 2π * (σz ⊗ σi + σi ⊗ σz)

w, v = eigen_decomp(H, 0.5, lvl=4)
w = 2π * w
ψ = (v[:, 1] + v[:, 2] + v[:, 3]) / sqrt(3)
ρ = ψ * ψ'
u = v' * ρ * v
w_ab = transpose(w) .- w
gm = γ.(w_ab)
sm = S.(w_ab)

# expected result
dρ, A_ij = ame_update_term(op, ρ, w, v, γ, S)

du = zeros(ComplexF64, (4, 4))
OpenQuantumBase.davies_update!(du, u, A_ij, gm, sm)
@test v * du * v' ≈ dρ atol = 1e-6 rtol = 1e-6

du = zeros(ComplexF64, (4, 4))
davies(du, u, w_ab, v, 0.5)
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
exp_effective_H = ame_trajectory_update_term(op, w, v, γ, S)
update_cache!(cache, davies, w_ab, v, 0.5)
@test v * cache * v' ≈ exp_effective_H atol = 1e-6 rtol = 1e-6

# test for DiffEqLiouvillian
ame_op = DiffEqLiouvillian(H, [davies], [], 4)
p = ODEParams(H, 2.0, (tf, t) -> t / tf)
exp_effective_H = ame_trajectory_update_term(op, w, v, γ, S) - 1.0im * H(0.5)
cache = zeros(ComplexF64, 4, 4)
update_cache!(cache, ame_op, p, 1)
@test cache ≈ exp_effective_H atol = 1e-6 rtol = 1e-6

hmat = H(0.5)
expected_drho = dρ - 1.0im * (hmat * ρ - ρ * hmat)
du = zeros(ComplexF64, (4, 4))
ame_op(du, ρ, p, 1)
@test du ≈ expected_drho atol = 1e-6 rtol = 1e-6

Random.seed!(1234)
jump_op = OpenQuantumBase.lindblad_jump(ame_op, ψ, p, 1)
exp_res = Complex{Float64}[6.672340269678421 + 0.0im 0.37611761184098264 + 0.0im 0.30757886113480604 + 0.0im -7.009918137704409 + 0.0im; 8.329733374366892 + 0.0im 0.46954431240210126 + 0.0im 0.38398070261603034 + 0.0im -8.751164764267996 + 0.0im; 9.465353222476072 + 0.0im 0.5335588272449766 + 0.0im 0.4363300501381911 + 0.0im -9.944239734825716 + 0.0im; 7.213269209357397 + 0.0im 0.4066095970732551 + 0.0im 0.33251438607759143 + 0.0im -7.578214632218711 + 0.0im]
@test size(jump_op) == (4, 4)
@test jump_op ≈ exp_res

# Dense Hamiltonian with size smaller than truncation levels
H = DenseHamiltonian(
    [(s) -> 1 - s, (s) -> s],
    [-standard_driver(4), random_ising(4)]
)

coupling = collective_coupling("Z", 4)
davies = OpenQuantumBase.DaviesGenerator(coupling, γ, S)
ame_op = DiffEqLiouvillian(H, [davies], [], 20)
p = ODEParams(H, 2.0, (tf, t) -> t / tf)
cache = zeros(ComplexF64, 16, 16)
update_cache!(cache, ame_op, p, 1)
@test cache != zeros(ComplexF64, 16, 16)

# Sparse Hamiltonian test
Hd = standard_driver(4; sp=true)
Hp = q_translate("-0.9ZZII+IZZI-0.9IIZZ"; sp=true)
H = SparseHamiltonian([(s) -> 1 - s, (s) -> s], [Hd, Hp])
op = 2π * q_translate("ZIII+IZII+IIZI+IIIZ")
coupling = ConstantCouplings(["ZIII+IZII+IIZI+IIIZ"])
davies = OpenQuantumBase.DaviesGenerator(coupling, γ, S)
w, v = eigen_decomp(H, 0.5, lvl=4)
w = 2π * real(w)

ψ = (v[:, 1] + v[:, 2] + v[:, 3]) / sqrt(3)
ρ = ψ * ψ'
u = v' * ρ * v
dρ, = ame_update_term(op, ρ, w, v, γ, S)
exp_effective_H =
    ame_trajectory_update_term(op, w, v, γ, S) - 1.0im * v * Diagonal(w) * v'

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
@test cache ≈ exp_effective_H atol = 1e-6 rtol = 1e-6

# test for adiabatc frame Hamiltonian
H = AdiabaticFrameHamiltonian((s) -> [0, s, 1 - s, 1], nothing)
hmat =  H(2.0, 0.4)
w = diag(hmat)
v = collect(Diagonal(ones(4)))
coupling = CustomCouplings([(s) -> s * (σx ⊗ σi + σi ⊗ σx) + (1 - s) * (σz ⊗ σi + σi ⊗ σz)])
davies = OpenQuantumBase.DaviesGenerator(coupling, γ, S)

ψ = (v[:, 1] + v[:, 2] + v[:, 3]) / sqrt(3)
ρ = ψ * ψ'

dρ, = ame_update_term(coupling(0.4)[1], ρ, w, v, γ, S)
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
γfun(w) = w >= 0 ? 1.0 : exp(-0.5)
cbath = CorrelatedBath(((1, 2), (2, 1)), spectrum=[(w) -> 0 γfun; γfun (w) -> 0])
D = OpenQuantumBase.davies_from_interactions(InteractionSet(Interaction(coupling, cbath)), 1:10, false, nothing)[1]
@test typeof(D) <: OpenQuantumBase.CorrelatedDaviesGenerator
du = zeros(ComplexF64, 2, 2)
ρ = [0.5 0;0 0.5]
ω = [1, 2]
ω =  ω' .- ω
D(du, ρ, ω, 0.5)
@test du ≈ [(1 - exp(-0.5)) * 0.5 0; 0 -(1 - exp(-0.5)) * 0.5]