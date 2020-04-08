using QTBase, Test

H = DenseHamiltonian([(x)->x], [σz])
u0 = PauliVec[1][1]

annealing = Annealing(H, u0)
@test annealing.H == H
