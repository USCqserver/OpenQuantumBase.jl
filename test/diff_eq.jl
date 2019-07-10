using QTBase, DifferentialEquations, Test

H = hamiltonian_factory([(x)->2.5], [σx])
sol = calculate_unitary(H, 2; reltol=1e-8, abstol=1e-8)
u_res = exp(-1.0im*5*0.5*σx)

@test isapprox(u_res, sol(0.5), rtol=1e-6, atol=1e-8)

u0 = [1.0, 0.0im]
sol = solve_schrodinger(H, u0, 2; reltol=1e-8, abstol=1e-8)
u_res = exp(-1.0im*5*0.5*σx) * u0
@test isapprox(u_res, sol(0.5), rtol=1e-6, atol=1e-8)
