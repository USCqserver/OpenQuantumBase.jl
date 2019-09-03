using QTBase, Test


H = DenseHamiltonian([(s)->1-s, (s)->s], [σx, σz])
dH(s) = (-real(σx) + real(σz))*2*π
coupling = ConstantCouplings(["Z"])

t_obj = project_to_lowlevel(H, dH, coupling, [0.0, 0.5, 1.0])

@test isapprox(
    t_obj.ev ./ 2 ./ π,
    [[-1.0, 1.0], [-0.707107, 0.707107], [-1.0, 1.0]],
    atol = 1e-6
)
@test t_obj.dθ ≈ [[0.5], [1.0], [0.5]]
@test get_dθ(t_obj) ≈ -[0.5, 1.0, 0.5]

@test isapprox(
    t_obj.op ./ 2 ./ π,
    [
     [[0 -1.0; -1.0 0]],
     [[-0.707107 -0.707107; -0.707107 0.707107]],
     [[-1.0 0.0; 0.0 1.0]]
    ],
    atol = 1e-6
)

H = SparseHamiltonian([(s)->1-s, (s)->s], [-standard_driver(2, sp=true) / 2, (spσz⊗spσi-0.1spσz⊗spσz) / 2])
dH(s) = standard_driver(2, sp=true) / 2 + (spσz⊗spσi-0.1spσz⊗spσz) / 2
coupling = ConstantCouplings(["ZI", "IZ"])

t_obj = project_to_lowlevel(H, dH, coupling, [0.0, 0.5, 1.0])

@test isapprox(
    t_obj.ev ./ π,
    [[-2.0, 0.0], [-1.2088723439378917, -0.20887234393789134], [-1.1, -0.9]],
    atol = 1e-6
)
