using QTBase, Test

H = DenseHamiltonian([(x)->x], [σz])

annealing = AnnealingParams(H, UnitTime(10))
exp_annealing = set_tf(annealing, 20)
@test exp_annealing.tf.t == 20.0

annealing = AnnealingParams(H, 10)
exp_annealing = set_tf(annealing, 20)
@test exp_annealing.tf == 20.0
