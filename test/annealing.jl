using QTBase, Test

H = DenseHamiltonian([(x)->x], [σz])
u0 = PauliVec[1][1]

@test_throws MethodError annealing = Annealing(H, u0, H)

annealing = AnnealingParams(H, UnitTime(10))
exp_annealing = set_tf(annealing, 20)
@test exp_annealing.tf.t == 20.0

annealing = LightAnnealingParams(UnitTime(10))
exp_annealing = set_tf(annealing, 20)
@test exp_annealing.tf.t == 20.0

annealing = AnnealingParams(H, 10)
exp_annealing = set_tf(annealing, 20)
@test exp_annealing.tf == 20.0

annealing = LightAnnealingParams(10)
exp_annealing = set_tf(annealing, 20)
@test exp_annealing.tf == 20.0
