using QTBase, Test
import LinearAlgebra: Diagonal


#================ Define  test functions ==================#
# calculate the AME linblad term using the formula in reference paper
function ame_update_term(op, w, v, γ, S)
    L_ij = Array{Array{Complex{Float64},2},2}(undef, (4, 4))
    A_ij = Array{Complex{Float64},2}(undef, (4, 4))
    for i = 1:4
        for j = 1:4
            A_ij[i, j] = v[:, i]' * op * v[:, j]
            L_ij[i, j] = v[:, i]' * op * v[:, j] * v[:, i] * v[:, j]'
        end
    end
    drho = []
    hls = []
    w_ab = Array{Float64,2}(undef, (4, 4))
    for i = 1:4
        for j = 1:4
            w_ab[i, j] = w[j] - w[i]
            if i != j
                T = L_ij[i, j]' * L_ij[i, j]
                push!(
                    drho,
                    γ(w[j] - w[i]) * (
                        L_ij[i, j] * rho * L_ij[i, j]' -
                        0.5 * (T * rho + rho * T)
                    ),
                )
                push!(hls, T * S(w[j] - w[i]))
            end
            T = L_ij[i, i]' * L_ij[j, j]
            push!(
                drho,
                γ(0) *
                (L_ij[i, i] * rho * L_ij[j, j]' - 0.5 * (T * rho + rho * T)),
            )
            push!(hls, T * S(0))
        end
    end
    drho = sum(drho)
    hls = sum(hls)
    drho = drho - 1.0im * (hls * rho - rho * hls)
    drho, A_ij
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


γ(x) = x >= 0 ? x + 1 : (1 - x) * exp(x)
S(x) = x + 0.1
#================= Define test functions ==================#

H = DenseHamiltonian(
    [(s) -> 1 - s, (s) -> s],
    [-standard_driver(2), (0.1 * σz ⊗ σi + 0.5 * σz ⊗ σz)],
)
coupling = ConstantCouplings(["ZI+IZ"])
davies_gen = DaviesGenerator(coupling, γ, S)
op = 2π * (σz ⊗ σi + σi ⊗ σz)

w, v = eigen_decomp(H, 0.5, level = 4)
w = 2π * w
state = (v[:, 1] + v[:, 2] + v[:, 3]) / sqrt(3)
rho = state * state'
u = v' * rho * v
w_ab = transpose(w) .- w
gm = γ.(w_ab)
sm = S.(w_ab)

# expected result
drho, A_ij = ame_update_term(op, w, v, γ, S)

du = zeros(ComplexF64, (4, 4))
QTBase.davies_update!(du, u, A_ij, gm, sm)
@test v * du * v' ≈ drho atol = 1e-6 rtol = 1e-6

du = zeros(ComplexF64, (4, 4))
davies_gen(du, u, w_ab, v, 1.0, 0.5)
@test v * du * v' ≈ drho atol = 1e-6 rtol = 1e-6
du = zeros(ComplexF64, (4, 4))
davies_gen(du, u, w_ab, v, UnitTime(1.0), 5)
@test v * du * v' ≈ drho atol = 1e-6 rtol = 1e-6

ame_op = AMETrajectoryOperator(H, davies_gen, 4)
exp_effective_H = ame_trajectory_update_term(op, w, v, γ, S) - 1.0im * H(0.5)
cache = zeros(ComplexF64, 4, 4)
update_cache!(cache, ame_op, 1.0, 0.5)
@test cache ≈ exp_effective_H atol = 1e-6 rtol = 1e-6


#TODO: Add test for the operator itself
jump_op = ame_jump(ame_op, state, 1.0, 0.5)
@test size(jump_op) == (4, 4)

# test for AMEDiffEqOperator
ame_op = AMEDiffEqOperator(H, davies_gen, 4)
hmat = H(0.5)
expected_drho = drho - 1.0im * (hmat * rho - rho * hmat)

du = zeros(ComplexF64, (4, 4))
ame_op(du, rho, ODEParams(2.0), 0.5)
@test du ≈ 2.0 * expected_drho atol = 1e-6 rtol = 1e-6

du = zeros(ComplexF64, (4, 4))
ame_op(du, rho, ODEParams(UnitTime(5)), 2.5)
@test du ≈ expected_drho atol = 1e-6 rtol = 1e-6

# Sparse Hamiltonian test
Hd = standard_driver(4; sp = true)
Hp = q_translate("-0.9ZZII+IZZI-0.9IIZZ"; sp = true)
H = SparseHamiltonian([(s) -> 1 - s, (s) -> s], [Hd, Hp])
op = 2π * q_translate("ZIII+IZII+IIZI+IIIZ")
coupling = ConstantCouplings(["ZIII+IZII+IIZI+IIIZ"])
davies_gen = DaviesGenerator(coupling, γ, S)
w, v = eigen_decomp(H, 0.5, level = 4, tol = 0.0)
w = 2π * real(w)

state = (v[:, 1] + v[:, 2] + v[:, 3]) / sqrt(3)
rho = state * state'
u = v' * rho * v
drho, = ame_update_term(op, w, v, γ, S)
exp_effective_H =
    ame_trajectory_update_term(op, w, v, γ, S) - 1.0im * v * Diagonal(w) * v'


ame_op = AMEDiffEqOperator(H, davies_gen, 4)
du = zeros(ComplexF64, (16, 16))
ame_op(du, rho, ODEParams(1.0), 0.5)
hmat = H(0.5)
@test isapprox(
    du,
    drho - 1.0im * (hmat * rho - rho * hmat),
    atol = 1e-6,
    rtol = 1e-6,
)

ame_op = AMETrajectoryOperator(H, davies_gen, 4)
cache = zeros(ComplexF64, 16, 16)
update_cache!(cache, ame_op, 1.0, 0.5)
@test cache ≈ exp_effective_H atol = 1e-6 rtol = 1e-6
