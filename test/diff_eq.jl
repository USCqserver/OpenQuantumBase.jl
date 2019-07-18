using QTBase, DifferentialEquations, Test

H = DenseHamiltonian([(x)->2.5], [σx])
u0 = [1.0, 0.0im]
test_annealing = Annealing(H, u0)
sol = solve_unitary(test_annealing, 2; reltol=1e-8, abstol=1e-8)
u_res = exp(-1.0im*5*0.5*σx)

@test isapprox(u_res, sol(0.5), rtol=1e-6, atol=1e-8)

sol = solve_schrodinger(test_annealing, 2; alg=Tsit5(), reltol=1e-8, abstol=1e-8)
u_res = exp(-1.0im*5*0.5*σx) * u0
@test isapprox(u_res, sol(0.5), rtol=1e-6, atol=1e-8)
