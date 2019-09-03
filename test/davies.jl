using QTBase, Test
import LinearAlgebra:eigen, Hermitian

function γ(x)
    if x==0
        return 1
    elseif x>0
        return x+1
    else
        return (1-x)*exp(x)
    end
end

function S(x)
    return x+0.1
end

H = DenseHamiltonian([(s)->1-s, (s)->s], [-standard_driver(2), (0.1*σz⊗σi + 0.5*σz⊗σz)])
coupling = ConstantCouplings(["ZI+IZ"])
davies_gen = QTBase.DaviesGenerator(coupling, γ, S)
op = 2π * (σz⊗σi + σi⊗σz)

w, v = eigen(Hermitian(H(0.5)))
state = (v[:, 1] + v[:, 2] + v[:, 3])/sqrt(3)
rho = state* state'
u = v'*rho*v
L_ij = Array{Array{Complex{Float64},2}, 2}(undef, (4, 4))
A_ij = Array{Complex{Float64}, 2}(undef, (4, 4))
for i in 1:4
    for j in 1:4
        A_ij[i,j] = v[:, i]' * op * v[:, j]
        L_ij[i,j] = v[:, i]' * op * v[:, j] * v[:, i] * v[:, j]'
    end
end
drho = []
hls= []
w_ab = Array{Float64, 2}(undef, (4,4))
for i in 1:4
    for j in 1:4
        w_ab[i,j] = w[j] - w[i]
        if i!=j
            T = L_ij[i,j]' * L_ij[i,j]
            push!(drho, γ(w[j]-w[i])*(L_ij[i,j]*rho*L_ij[i,j]'-0.5*(T*rho+rho*T)))
            push!(hls, T*S(w[j]-w[i]))
        end
        T = L_ij[i,i]' * L_ij[j,j]
        push!(drho, γ(0)*(L_ij[i,i]*rho*L_ij[j,j]'-0.5*(T*rho+rho*T)))
        push!(hls, T*S(0))
    end
end
drho = sum(drho)
hls = sum(hls)
drho = drho -1.0im * (hls*rho - rho*hls)

gm = γ.(w_ab)
sm = S.(w_ab)
du = zeros(ComplexF64, (4,4))
QTBase.adiabatic_me_update!(du, u, A_ij, gm, sm)
@test isapprox(v * du * v', drho, atol=1e-6, rtol=1e-6)


du = zeros(ComplexF64, (4,4))
davies_gen(du, u, w_ab, v, 1.0, 1.0)
@test isapprox(v * du * v', drho, atol=1e-6, rtol=1e-6)
