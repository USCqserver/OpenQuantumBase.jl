using QTBase, Test, Random
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
davies = DaviesGenerator(coupling, γ, S)
op = 2π * (σz ⊗ σi + σi ⊗ σz)

w, v = eigen_decomp(H, 0.5, lvl = 4)
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
davies(du, u, w_ab, v, 0.5)
@test v * du * v' ≈ drho atol = 1e-6 rtol = 1e-6

cache = zeros(ComplexF64, (4, 4))
exp_effective_H = ame_trajectory_update_term(op, w, v, γ, S)
update_cache!(cache, davies, w_ab, v, 0.5)
@test v * cache * v' ≈ exp_effective_H atol = 1e-6 rtol = 1e-6

# test for AMEOperator
ame_op = AMEOperator(H, davies, 4)
p = ODEParams(H, 2.0, (tf, t)->t/tf)
exp_effective_H = ame_trajectory_update_term(op, w, v, γ, S) - 1.0im * H(0.5)
cache = zeros(ComplexF64, 4, 4)
update_cache!(cache, ame_op, p, 1)
@test cache ≈ exp_effective_H atol = 1e-6 rtol = 1e-6

hmat = H(0.5)
expected_drho = drho - 1.0im * (hmat * rho - rho * hmat)
du = zeros(ComplexF64, (4, 4))
ame_op(du, rho, p, 1)
@test du ≈ expected_drho atol = 1e-6 rtol = 1e-6

Random.seed!(1234)
jump_op = ame_jump(ame_op, state, p, 1)
exp_res = Complex{Float64}[6.672340269678421 + 0.0im 0.37611761184098264 + 0.0im 0.30757886113480604 + 0.0im -7.009918137704409 + 0.0im; 8.329733374366892 + 0.0im 0.46954431240210126 + 0.0im 0.38398070261603034 + 0.0im -8.751164764267996 + 0.0im; 9.465353222476072 + 0.0im 0.5335588272449766 + 0.0im 0.4363300501381911 + 0.0im -9.944239734825716 + 0.0im; 7.213269209357397 + 0.0im 0.4066095970732551 + 0.0im 0.33251438607759143 + 0.0im -7.578214632218711 + 0.0im]
@test size(jump_op) == (4, 4)
@test jump_op ≈ exp_res

# Sparse Hamiltonian test
Hd = standard_driver(4; sp = true)
Hp = q_translate("-0.9ZZII+IZZI-0.9IIZZ"; sp = true)
H = SparseHamiltonian([(s) -> 1 - s, (s) -> s], [Hd, Hp])
op = 2π * q_translate("ZIII+IZII+IIZI+IIIZ")
coupling = ConstantCouplings(["ZIII+IZII+IIZI+IIIZ"])
davies = DaviesGenerator(coupling, γ, S)
w, v = eigen_decomp(H, 0.5, lvl = 4)
w = 2π * real(w)

state = (v[:, 1] + v[:, 2] + v[:, 3]) / sqrt(3)
rho = state * state'
u = v' * rho * v
drho, = ame_update_term(op, w, v, γ, S)
exp_effective_H =
    ame_trajectory_update_term(op, w, v, γ, S) - 1.0im * v * Diagonal(w) * v'

ame_op = AMEOperator(H, davies, 4)
du = zeros(ComplexF64, (16, 16))
ame_op(du, rho, p, 1)
hmat = H(0.5)
@test isapprox(
    du,
    drho - 1.0im * (hmat * rho - rho * hmat),
    atol = 1e-6,
    rtol = 1e-6,
)

cache = zeros(ComplexF64, 16, 16)
update_cache!(cache, ame_op, p, 1)
@test cache ≈ exp_effective_H atol = 1e-6 rtol = 1e-6
