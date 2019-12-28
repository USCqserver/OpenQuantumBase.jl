using QTBase, Test
import LinearAlgebra: eigen, Hermitian


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
                    γ(w[j] - w[i]) * (L_ij[i, j] * rho * L_ij[i, j]' - 0.5 * (T * rho + rho * T)),
                )
                push!(hls, T * S(w[j] - w[i]))
            end
            T = L_ij[i, i]' * L_ij[j, j]
            push!(drho, γ(0) * (L_ij[i, i] * rho * L_ij[j, j]' - 0.5 * (T * rho + rho * T)))
            push!(hls, T * S(0))
        end
    end
    drho = sum(drho)
    hls = sum(hls)
    drho = drho - 1.0im * (hls * rho - rho * hls)
    drho, A_ij
end


γ(x) = x >= 0 ? x + 1 : (1 - x) * exp(x)
S(x) = x + 0.1
H = DenseHamiltonian(
    [(s) -> 1 - s, (s) -> s],
    [-standard_driver(2), (0.1 * σz ⊗ σi + 0.5 * σz ⊗ σz)],
)
coupling = ConstantCouplings(["ZI+IZ"])
davies_gen = QTBase.DaviesGenerator(coupling, γ, S)
op = 2π * (σz ⊗ σi + σi ⊗ σz)

H(0.5)
w, v = QTBase.ode_eigen_decomp(H, 4)
state = (v[:, 1] + v[:, 2] + v[:, 3]) / sqrt(3)
rho = state * state'
u = v' * rho * v
drho, A_ij = ame_update_term(op, w, v, γ, S)

w_ab = transpose(w) .- w
gm = γ.(w_ab)
sm = S.(w_ab)
du = zeros(ComplexF64, (4, 4))
QTBase.adiabatic_me_update!(du, u, A_ij, gm, sm)
@test isapprox(v * du * v', drho, atol = 1e-6, rtol = 1e-6)


du = zeros(ComplexF64, (4, 4))
davies_gen(du, u, w_ab, v, 1.0, 0.5)
@test isapprox(v * du * v', drho, atol = 1e-6, rtol = 1e-6)


#TODO: Add test for the operator itself
jump_op = QTBase.ame_jump(w, v, state, davies_gen, 1.0, 0.5)
@test size(jump_op) == (4, 4)

ame_op = AMEDiffEqOperator(H, davies_gen)
du = zeros(ComplexF64, (4, 4))
ame_op(du, rho, (tf = 1.0,), 0.5)
hmat = H(0.5)
@test isapprox(du, drho - 1.0im * (hmat * rho - rho * hmat), atol = 1e-6, rtol = 1e-6)


Hd = standard_driver(4; sp = true)
Hp = q_translate("-0.9ZZII+IZZI-0.9IIZZ"; sp = true)
H = SparseHamiltonian([(s) -> 1 - s, (s) -> s], [Hd, Hp])
op = 2π * q_translate("ZIII+IZII+IIZI+IIIZ")
coupling = ConstantCouplings(["ZIII+IZII+IIZI+IIIZ"])
davies_gen = QTBase.DaviesGenerator(coupling, γ, S)
H(0.5)
w, v = QTBase.ode_eigen_decomp(H, 4)

state = (v[:, 1] + v[:, 2] + v[:, 3]) / sqrt(3)
rho = state * state'
u = v' * rho * v
drho, = ame_update_term(op, w, v, γ, S)

du = zeros(ComplexF64, (4, 4))
w_ab = transpose(w) .- w
davies_gen(du, u, w_ab, v, 1.0, 0.5)
@test isapprox(v * du * v', drho, atol = 1e-6, rtol = 1e-6)

ame_op = AMEDiffEqOperator(H, davies_gen; lvl = 4)
du = zeros(ComplexF64, (16, 16))
ame_op(du, rho, (tf = 1.0,), 0.5)
hmat = H(0.5)
@test isapprox(du, drho - 1.0im * (hmat * rho - rho * hmat), atol = 1e-6, rtol = 1e-6)
