using QTBase, Test

dθ= (s)->π/2
gap = (s)-> 1.0
H = hamiltonian_factory([gap], [-σz], [dθ], [-σx])
u0 = [1.0+0.0im, 0]
normal_anneal = annealing_factory(H, u0)
piecewise_anneal = annealing_factory(H, u0, 0.5, 1.0)

@test piecewise_anneal.ode_problem.p.stops == [0.5, 1.5]
